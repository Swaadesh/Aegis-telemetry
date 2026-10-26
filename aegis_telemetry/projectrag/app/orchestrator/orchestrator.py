import time
import uuid

from app.config.settings import Settings
from app.context.base import ContextResult, ContextRetriever, ContextRetrieverError
from app.models.schemas import AIRequest, AIResponse, ErrorResponse
from app.orchestrator.model_router import ModelRouter
from app.prompts.prompt_builder import PromptBuilder
from app.providers.base import ProviderError, ProviderTimeoutError
from app.services.pending_search_store import PendingSearchStore
from app.services.response_formatter import format_response
from app.services.response_lock_store import ResponseLockStore
from app.utils.logging import log_request


class OutOfScopeError(Exception):
    """Raised when document retrieval found nothing relevant.

    The original prompt is held in PendingSearchStore under request_id;
    the caller must resolve it via POST /api/v1/websearch-confirm.
    """

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        super().__init__(
            f"Request {request_id} has no document context; awaiting web-search confirmation"
        )


class ResponseLockedError(Exception):
    """Raised when a response exceeds settings.response_token_limit.

    The real response is held in the ResponseLockStore under request_id;
    the caller must unlock it via POST /api/v1/unlock.
    """

    def __init__(self, request_id: str, attempts_remaining: int) -> None:
        self.request_id = request_id
        self.attempts_remaining = attempts_remaining
        super().__init__(f"Response {request_id} locked pending admin unlock")


class AIOrchestrationError(Exception):
    """Raised when the orchestrator exhausts retries or hits a fatal error."""

    def __init__(self, request_id: str, latency_ms: float, error_type: str, message: str) -> None:
        self.request_id = request_id
        self.latency_ms = latency_ms
        self.error_type = error_type
        self.message = message
        super().__init__(message)

    def to_response(self) -> ErrorResponse:
        return ErrorResponse(
            request_id=self.request_id,
            error_type=self.error_type,
            message=self.message,
            latency_ms=self.latency_ms,
        )


class AIOrchestrator:
    """Central entry point for AI requests.

    The API layer never calls a provider directly — every request flows
    through here: retrieve context (optional) -> build prompt -> route to a
    provider (with retry) -> validate and format the response -> log ->
    return. Model selection and context retrieval are both delegated to
    interfaces (ModelRouter, ContextRetriever) so Phase 2 additions
    (multi-model routing, a real RAG module) don't change this class.
    """

    def __init__(
        self,
        router: ModelRouter,
        settings: Settings,
        prompt_builder: PromptBuilder | None = None,
        context_retriever: ContextRetriever | None = None,
        lock_store: ResponseLockStore | None = None,
        pending_search_store: PendingSearchStore | None = None,
        web_retriever: ContextRetriever | None = None,
    ) -> None:
        self._router = router
        self._settings = settings
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._context_retriever = context_retriever
        self._lock_store = lock_store or ResponseLockStore()
        self._pending_search_store = pending_search_store or PendingSearchStore(
            ttl_seconds=settings.pending_search_ttl_seconds
        )
        self._web_retriever = web_retriever

    async def handle(
        self,
        request: AIRequest,
        user_id: str | None = None,
        role: str | None = None,
    ) -> AIResponse:
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        context, retrieval_latency_ms = await self._retrieve_context(request.prompt)

        if context is not None and not context.documents:
            self._pending_search_store.put(
                request_id,
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                user_id=user_id,
                role=role,
            )
            log_request(
                request_id=request_id,
                model=self._settings.model_name,
                latency_ms=(time.perf_counter() - start) * 1000,
                status="out_of_scope",
                user_id=user_id,
                role=role,
                **self._retrieval_log_fields(context, retrieval_latency_ms),
            )
            raise OutOfScopeError(request_id)

        system_prompt, user_prompt = self._prompt_builder.build(
            request.prompt, request.system_prompt, context
        )
        return await self._generate_and_lock(
            request_id,
            start,
            system_prompt,
            user_prompt,
            context,
            retrieval_latency_ms,
            user_id,
            role,
            source="document",
        )

    async def confirm_web_search(
        self,
        request_id: str,
        confirm: bool,
        user_id: str | None = None,
        role: str | None = None,
    ) -> AIResponse | None:
        """Resolve a pending out-of-scope request.

        Returns None if declined or if the pending entry is missing/expired
        (callers needing to distinguish those two cases should peek/check
        the store themselves before calling, as app/api/routes.py does).
        """
        entry = self._pending_search_store.evict(request_id)
        if entry is None or not confirm:
            return None

        start = time.perf_counter()
        context: ContextResult | None = None
        retrieval_latency_ms: float | None = None
        if self._web_retriever is not None:
            retrieval_start = time.perf_counter()
            try:
                context = await self._web_retriever.retrieve(entry.prompt)
            except ContextRetrieverError as exc:
                latency_ms = (time.perf_counter() - start) * 1000
                raise AIOrchestrationError(
                    request_id, latency_ms, "web_search_failed", str(exc)
                ) from exc
            retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

        system_prompt, user_prompt = self._prompt_builder.build(
            entry.prompt, entry.system_prompt, context
        )
        return await self._generate_and_lock(
            request_id,
            start,
            system_prompt,
            user_prompt,
            context,
            retrieval_latency_ms,
            entry.user_id,
            entry.role,
            source="web",
        )

    async def _generate_and_lock(
        self,
        request_id: str,
        start: float,
        system_prompt: str,
        user_prompt: str,
        context: ContextResult | None,
        retrieval_latency_ms: float | None,
        user_id: str | None,
        role: str | None,
        source: str,
    ) -> AIResponse:
        provider = await self._router.route()

        attempts = self._settings.max_retries + 1
        last_error: ProviderError | None = None

        for attempt in range(attempts):
            try:
                raw = await provider.generate(system_prompt, user_prompt)
                latency_ms = (time.perf_counter() - start) * 1000
                response = format_response(
                    request_id, latency_ms, raw, context, source=source  # type: ignore[arg-type]
                )
                total_tokens = response.token_usage.total_tokens
                is_locked = (
                    total_tokens is not None and total_tokens > self._settings.response_token_limit
                )
                log_request(
                    request_id=request_id,
                    model=response.model,
                    latency_ms=latency_ms,
                    status="locked" if is_locked else "success",
                    user_id=user_id,
                    role=role,
                    prompt_tokens=response.token_usage.prompt_tokens,
                    completion_tokens=response.token_usage.completion_tokens,
                    total_tokens=response.token_usage.total_tokens,
                    threshold=self._settings.response_token_limit if is_locked else None,
                    **self._retrieval_log_fields(context, retrieval_latency_ms),
                )
                if is_locked:
                    self._lock_store.lock(request_id, response, self._settings.unlock_max_attempts)
                    raise ResponseLockedError(request_id, self._settings.unlock_max_attempts)
                return response
            except ProviderTimeoutError as exc:
                last_error = exc
                continue  # retry on timeout
            except ProviderError as exc:
                last_error = exc
                break  # quota/API errors are not retried
            except ValueError as exc:
                # e.g. empty response from format_response — not retryable
                latency_ms = (time.perf_counter() - start) * 1000
                log_request(
                    request_id=request_id,
                    model=self._settings.model_name,
                    latency_ms=latency_ms,
                    status="error",
                    user_id=user_id,
                    role=role,
                    error_type="invalid_response",
                    **self._retrieval_log_fields(context, retrieval_latency_ms),
                )
                raise AIOrchestrationError(request_id, latency_ms, "invalid_response", str(exc)) from exc

        latency_ms = (time.perf_counter() - start) * 1000
        error_type = type(last_error).__name__ if last_error else "unknown_error"
        message = str(last_error) if last_error else "AI provider failed with no further detail"
        log_request(
            request_id=request_id,
            model=self._settings.model_name,
            latency_ms=latency_ms,
            status="error",
            user_id=user_id,
            role=role,
            error_type=error_type,
            **self._retrieval_log_fields(context, retrieval_latency_ms),
        )
        raise AIOrchestrationError(request_id, latency_ms, error_type, message) from last_error

    async def _retrieve_context(
        self, query: str
    ) -> tuple[ContextResult | None, float | None]:
        if self._context_retriever is None:
            return None, None

        start = time.perf_counter()
        try:
            context = await self._context_retriever.retrieve(query)
        except ContextRetrieverError:
            return None, (time.perf_counter() - start) * 1000
        return context, (time.perf_counter() - start) * 1000

    @staticmethod
    def _retrieval_log_fields(
        context: ContextResult | None, retrieval_latency_ms: float | None
    ) -> dict:
        if context is None:
            return {
                "retrieval_latency_ms": retrieval_latency_ms,
                "documents_retrieved": None,
                "chunk_count": None,
                "citation_ids": None,
                "search_score": None,
            }
        scores = [doc.score for doc in context.documents]
        return {
            "retrieval_latency_ms": retrieval_latency_ms,
            "documents_retrieved": len(context.documents),
            "chunk_count": len(context.documents) or None,
            "citation_ids": context.citations,
            "search_score": max(scores) if scores else None,
        }

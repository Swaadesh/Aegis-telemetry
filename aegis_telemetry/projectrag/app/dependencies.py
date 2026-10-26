from functools import lru_cache

from app.config.settings import Settings, get_settings
from app.context.base import ContextRetriever
from app.context.pdf_retriever import PDFContextRetriever
from app.context.tavily_retriever import TavilyContextRetriever
from app.orchestrator.model_router import ModelRouter
from app.orchestrator.orchestrator import AIOrchestrator
from app.providers.gemini.gemini_provider import GeminiProvider
from app.services.pending_search_store import PendingSearchStore
from app.services.response_lock_store import ResponseLockStore


@lru_cache
def get_context_retriever() -> ContextRetriever:
    """Single shared retriever instance for the app's lifetime.

    Currently wired to PDFContextRetriever — a DEMO-ONLY keyword-search
    retriever over TokenLens_Telemetry_and_SpendGuard.pdf (see
    app/context/pdf_retriever.py), so the team can ask real questions
    about the spec from the terminal. This is NOT the real RAG module.

    This is the ONLY line that should change when Atif's real RAG module
    (chunking, embeddings, retrieval, ranking) replaces it — it must first
    pass the Step 10 contract test in
    tests/test_context_retriever_contract.py. To go back to the
    zero-dependency mock for orchestration-only testing, swap this for
    `from app.context.mock_retriever import MockContextRetriever; return
    MockContextRetriever()`.
    """
    settings = get_settings()
    return PDFContextRetriever(
        settings.knowledge_base_pdf_path, min_score=settings.retrieval_min_score
    )


@lru_cache
def get_response_lock_store() -> ResponseLockStore:
    """Single shared in-memory store for token-limit-locked responses."""
    return ResponseLockStore()


@lru_cache
def get_pending_search_store() -> PendingSearchStore:
    """Single shared in-memory store for prompts awaiting a web-search
    confirmation decision (see app/orchestrator/orchestrator.py::OutOfScopeError)."""
    settings = get_settings()
    return PendingSearchStore(ttl_seconds=settings.pending_search_ttl_seconds)


@lru_cache
def get_tavily_retriever() -> ContextRetriever:
    """Opt-in web search fallback, only invoked after a user confirms via
    POST /api/v1/websearch-confirm — see app/context/tavily_retriever.py."""
    settings = get_settings()
    return TavilyContextRetriever(
        api_key=settings.tavily_api_key, max_results=settings.tavily_max_results
    )


@lru_cache
def get_orchestrator() -> AIOrchestrator:
    """Single shared orchestrator instance for the app's lifetime.

    Swapping/adding providers later means changing only this wiring, not
    the orchestrator or API layer.
    """
    settings: Settings = get_settings()
    provider = GeminiProvider(settings)
    router = ModelRouter({settings.model_name: provider}, settings)
    return AIOrchestrator(
        router,
        settings,
        context_retriever=get_context_retriever(),
        lock_store=get_response_lock_store(),
        pending_search_store=get_pending_search_store(),
        web_retriever=get_tavily_retriever(),
    )

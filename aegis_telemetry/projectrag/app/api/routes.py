from fastapi import APIRouter, Depends, Header
from fastapi.responses import JSONResponse

from app.config.settings import Settings, get_settings
from app.dependencies import (
    get_orchestrator,
    get_pending_search_store,
    get_response_lock_store,
)
from app.models.schemas import (
    AIRequest,
    AIResponse,
    LockedResponse,
    OutOfScopeResponse,
    UnlockRequest,
    WebSearchConfirmRequest,
)
from app.orchestrator.orchestrator import (
    AIOrchestrationError,
    AIOrchestrator,
    OutOfScopeError,
    ResponseLockedError,
)
from app.services.pending_search_store import PendingSearchStore
from app.services.response_lock_store import ResponseLockStore
from app.utils.logging import log_lock_resolution, log_websearch_resolution

router = APIRouter(prefix="/api/v1")


@router.post("/generate", response_model=None)
async def generate(
    request: AIRequest,
    orchestrator: AIOrchestrator = Depends(get_orchestrator),
    x_user_id: str | None = Header(default=None),
    x_user_role: str | None = Header(default=None),
) -> AIResponse | JSONResponse:
    """`X-User-Id`/`X-User-Role` are forwarded by the gateway, which already
    authenticated the caller. This service trusts them as upstream-asserted
    headers the same way it would trust any other gateway-forwarded value —
    it does not validate, decode, or authenticate them (auth stays out of
    scope per CLAUDE.md).
    """
    try:
        return await orchestrator.handle(request, user_id=x_user_id, role=x_user_role)
    except AIOrchestrationError as exc:
        return JSONResponse(status_code=502, content=exc.to_response().model_dump())
    except ResponseLockedError as exc:
        locked = LockedResponse(
            request_id=exc.request_id,
            message="Token limit crossed. Enter admin secret key to view this response.",
            attempts_remaining=exc.attempts_remaining,
        )
        return JSONResponse(status_code=200, content=locked.model_dump())
    except OutOfScopeError as exc:
        out_of_scope = OutOfScopeResponse(
            request_id=exc.request_id,
            message=(
                "No answer found in the knowledge base. "
                "POST /api/v1/websearch-confirm with confirm=true to search the web instead."
            ),
        )
        return JSONResponse(status_code=200, content=out_of_scope.model_dump())


@router.post("/websearch-confirm", response_model=None)
async def websearch_confirm(
    body: WebSearchConfirmRequest,
    orchestrator: AIOrchestrator = Depends(get_orchestrator),
    pending_store: PendingSearchStore = Depends(get_pending_search_store),
    x_user_id: str | None = Header(default=None),
    x_user_role: str | None = Header(default=None),
) -> AIResponse | JSONResponse:
    """Resolve a pending out-of-scope request: search the web (Tavily) and
    generate, or decline. See OutOfScopeError / AIOrchestrator.confirm_web_search."""
    pending = pending_store.peek(body.request_id)
    if pending is None:
        return JSONResponse(
            status_code=404,
            content={
                "message": "No pending web-search request for this id (expired or already resolved)."
            },
        )

    if not body.confirm:
        pending_store.evict(body.request_id)
        log_websearch_resolution(
            request_id=body.request_id, event="declined", user_id=x_user_id, role=x_user_role
        )
        return JSONResponse(
            status_code=200,
            content={
                "request_id": body.request_id,
                "status": "declined",
                "message": "OK, not searching the web.",
            },
        )

    try:
        response = await orchestrator.confirm_web_search(
            body.request_id, confirm=True, user_id=x_user_id, role=x_user_role
        )
    except AIOrchestrationError as exc:
        return JSONResponse(status_code=502, content=exc.to_response().model_dump())
    except ResponseLockedError as exc:
        log_websearch_resolution(
            request_id=body.request_id, event="confirmed", user_id=x_user_id, role=x_user_role
        )
        locked = LockedResponse(
            request_id=exc.request_id,
            message="Token limit crossed. Enter admin secret key to view this response.",
            attempts_remaining=exc.attempts_remaining,
            source="web",
        )
        return JSONResponse(status_code=200, content=locked.model_dump())

    assert response is not None  # confirm=True and pending existed, so this always succeeds or raises
    log_websearch_resolution(
        request_id=body.request_id, event="confirmed", user_id=x_user_id, role=x_user_role
    )
    return response


@router.post("/unlock", response_model=None)
async def unlock(
    body: UnlockRequest,
    settings: Settings = Depends(get_settings),
    lock_store: ResponseLockStore = Depends(get_response_lock_store),
    x_user_id: str | None = Header(default=None),
    x_user_role: str | None = Header(default=None),
) -> AIResponse | JSONResponse:
    """Demo-grade guardrail, not real authentication — see
    app/services/response_lock_store.py and settings.admin_unlock_key."""
    entry = lock_store.peek(body.request_id)
    if entry is None:
        return JSONResponse(
            status_code=404,
            content={"message": "No locked response found for this request (expired or already resolved)."},
        )

    if body.admin_key == settings.admin_unlock_key:
        response = lock_store.evict(body.request_id)
        assert response is not None
        log_lock_resolution(
            request_id=body.request_id,
            event="unlocked",
            user_id=x_user_id,
            role=x_user_role,
        )
        return response

    attempts_remaining = lock_store.decrement(body.request_id)
    if attempts_remaining > 0:
        log_lock_resolution(
            request_id=body.request_id,
            event="attempt_failed",
            user_id=x_user_id,
            role=x_user_role,
            attempts_remaining=attempts_remaining,
        )
        return JSONResponse(
            status_code=401,
            content={"message": f"Attempt failed. {attempts_remaining} attempt(s) remaining."},
        )
    log_lock_resolution(
        request_id=body.request_id,
        event="blocked",
        user_id=x_user_id,
        role=x_user_role,
        attempts_remaining=0,
    )
    return JSONResponse(
        status_code=403,
        content={"message": "Sorry, token limit crossed. Reset your admin secret key or contact admin."},
    )


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

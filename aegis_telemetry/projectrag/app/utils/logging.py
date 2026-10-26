import json
import logging
import sys
from datetime import UTC, datetime

logger = logging.getLogger("aegis.ai")
logger.setLevel(logging.INFO)

if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)


def log_request(
    *,
    request_id: str,
    model: str,
    latency_ms: float,
    status: str,
    user_id: str | None = None,
    role: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    error_type: str | None = None,
    retrieval_latency_ms: float | None = None,
    documents_retrieved: int | None = None,
    chunk_count: int | None = None,
    citation_ids: list[str] | None = None,
    search_score: float | None = None,
    threshold: int | None = None,
) -> None:
    """Emit one structured JSON log line per AI request.

    Consumed by the telemetry dashboards (see phase2.txt Step 7/8) — the
    shape here is a commitment, so add fields rather than renaming/removing
    existing ones. RAG fields default to None/empty for non-RAG requests.
    `threshold` is set when status="locked", recording the limit that was
    crossed. `status="out_of_scope"` records a request where document
    retrieval found nothing relevant, pending a web-search decision.
    """
    line = {
        "timestamp": datetime.now(UTC).isoformat(),
        "request_id": request_id,
        "user_id": user_id,
        "role": role,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_ms": round(latency_ms, 1),
        "status": status,
        "error_type": error_type,
        "retrieval_latency_ms": (
            round(retrieval_latency_ms, 1) if retrieval_latency_ms is not None else None
        ),
        "documents_retrieved": documents_retrieved,
        "chunk_count": chunk_count,
        "citation_ids": citation_ids or [],
        "search_score": search_score,
        "threshold": threshold,
    }
    logger.info(json.dumps(line))


def log_lock_resolution(
    *,
    request_id: str,
    event: str,
    user_id: str | None = None,
    role: str | None = None,
    attempts_remaining: int | None = None,
) -> None:
    """Emit one structured JSON log line per admin-unlock resolution.

    `event` is one of "unlocked", "attempt_failed", "blocked" — the outcome
    of a POST /api/v1/unlock call against a response the orchestrator
    previously locked (see ResponseLockedError). Pairs with the "locked"
    log_request line for the same request_id to give the dashboard the full
    lifecycle: who hit the limit, what threshold, and how it was resolved.
    """
    line = {
        "timestamp": datetime.now(UTC).isoformat(),
        "request_id": request_id,
        "event": event,
        "user_id": user_id,
        "role": role,
        "attempts_remaining": attempts_remaining,
    }
    logger.info(json.dumps(line))


def log_websearch_resolution(
    *,
    request_id: str,
    event: str,
    user_id: str | None = None,
    role: str | None = None,
) -> None:
    """Emit one structured JSON log line per web-search-confirmation outcome.

    `event` is one of "confirmed", "declined", "expired" — pairs with the
    "out_of_scope" log_request line for the same request_id to give the
    dashboard the full lifecycle: when document retrieval came up empty,
    and whether/how the user chose to fall back to a web search.
    """
    line = {
        "timestamp": datetime.now(UTC).isoformat(),
        "request_id": request_id,
        "event": event,
        "user_id": user_id,
        "role": role,
    }
    logger.info(json.dumps(line))

from dataclasses import dataclass

from app.models.schemas import AIResponse


@dataclass
class _LockedEntry:
    response: AIResponse
    attempts_remaining: int


class ResponseLockStore:
    """In-memory holding area for responses gated behind the admin unlock key.

    Single-process, in-memory by design — entries are lost on restart and
    not shared across worker processes. Acceptable for the current
    single-instance Phase 1 demo scope; a multi-worker deployment would need
    a shared store (e.g. Redis) instead.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _LockedEntry] = {}

    def lock(self, request_id: str, response: AIResponse, attempts: int) -> None:
        self._entries[request_id] = _LockedEntry(response=response, attempts_remaining=attempts)

    def peek(self, request_id: str) -> _LockedEntry | None:
        return self._entries.get(request_id)

    def decrement(self, request_id: str) -> int:
        """Decrement attempts_remaining and return the new value.

        Evicts the entry once attempts are exhausted.
        """
        entry = self._entries.get(request_id)
        if entry is None:
            return 0
        entry.attempts_remaining -= 1
        if entry.attempts_remaining <= 0:
            del self._entries[request_id]
            return 0
        return entry.attempts_remaining

    def evict(self, request_id: str) -> AIResponse | None:
        entry = self._entries.pop(request_id, None)
        return entry.response if entry else None

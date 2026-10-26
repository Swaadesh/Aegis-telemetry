import time
from dataclasses import dataclass


@dataclass
class PendingEntry:
    prompt: str
    system_prompt: str | None
    user_id: str | None
    role: str | None
    created_at: float


class PendingSearchStore:
    """In-memory holding area for prompts awaiting a web-search confirmation.

    Same single-process, in-memory-only scope as ResponseLockStore — entries
    are lost on restart and not shared across worker processes, which is
    acceptable for this demo-grade Phase 1 exception. Expiry is checked
    lazily on peek/evict; there is no background sweep.
    """

    def __init__(self, ttl_seconds: float = 300) -> None:
        self._ttl_seconds = ttl_seconds
        self._entries: dict[str, PendingEntry] = {}

    def put(
        self,
        request_id: str,
        *,
        prompt: str,
        system_prompt: str | None,
        user_id: str | None,
        role: str | None,
    ) -> None:
        self._entries[request_id] = PendingEntry(
            prompt=prompt,
            system_prompt=system_prompt,
            user_id=user_id,
            role=role,
            created_at=time.monotonic(),
        )

    def peek(self, request_id: str) -> PendingEntry | None:
        entry = self._entries.get(request_id)
        if entry is None:
            return None
        if time.monotonic() - entry.created_at > self._ttl_seconds:
            del self._entries[request_id]
            return None
        return entry

    def evict(self, request_id: str) -> PendingEntry | None:
        entry = self.peek(request_id)
        if entry is None:
            return None
        del self._entries[request_id]
        return entry

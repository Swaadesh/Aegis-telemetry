import time

from app.services.pending_search_store import PendingSearchStore


def test_put_then_peek_returns_entry() -> None:
    store = PendingSearchStore(ttl_seconds=300)
    store.put("req-1", prompt="what is X?", system_prompt=None, user_id="u1", role="analyst")

    entry = store.peek("req-1")

    assert entry is not None
    assert entry.prompt == "what is X?"
    assert entry.user_id == "u1"
    assert entry.role == "analyst"


def test_peek_unknown_id_returns_none() -> None:
    store = PendingSearchStore(ttl_seconds=300)
    assert store.peek("does-not-exist") is None


def test_evict_removes_entry() -> None:
    store = PendingSearchStore(ttl_seconds=300)
    store.put("req-1", prompt="q", system_prompt=None, user_id=None, role=None)

    evicted = store.evict("req-1")

    assert evicted is not None
    assert evicted.prompt == "q"
    assert store.peek("req-1") is None


def test_evict_unknown_id_returns_none() -> None:
    store = PendingSearchStore(ttl_seconds=300)
    assert store.evict("does-not-exist") is None


def test_entry_expires_after_ttl() -> None:
    store = PendingSearchStore(ttl_seconds=0.05)
    store.put("req-1", prompt="q", system_prompt=None, user_id=None, role=None)

    time.sleep(0.1)

    assert store.peek("req-1") is None
    assert store.evict("req-1") is None

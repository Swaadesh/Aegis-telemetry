# Context Retrieval Contract — v1

This is the versioned interface Atif's real RAG module must implement. It
maps 1:1 onto the `ContextRetriever`/`RetrievedDocument`/`ContextResult`
types in `app/context/base.py` — those Python types ARE the contract; this
doc makes it explicit and shareable outside the codebase.

Any implementation must pass `tests/test_context_retriever_contract.py`
before being wired into `app/dependencies.py`.

## Interface

```python
class ContextRetriever(ABC):
    async def retrieve(self, query: str, top_k: int = 5) -> ContextResult: ...
```

## Input

| Field     | Type   | Required | Notes                                   |
|-----------|--------|----------|------------------------------------------|
| query     | string | yes      | The user's prompt/question                |
| user_id   | string | no       | Forwarded from `X-User-Id`, for scoping   |
| tenant    | string | no       | Multi-tenant isolation, if applicable     |
| top_k     | int    | no       | Max documents to return (default 5)       |

Conceptually:

```json
{
  "query": "string",
  "user_id": "string",
  "tenant": "string",
  "top_k": 5
}
```

Note: the current `ContextRetriever.retrieve(query, top_k)` signature only
carries `query`/`top_k`. `user_id`/`tenant` are documented here for the
retrieval backend's own scoping/access-control needs (owned by Abdullah's
vector DB layer) — they are not yet threaded through the Python interface
in Phase 2 and can be added as keyword args without breaking callers.

## Output — `ContextResult`

```json
{
  "documents": [
    {
      "text": "string",
      "source": "string",
      "score": 0.94,
      "metadata": {}
    }
  ],
  "citations": ["string"],
  "confidence": 0.94
}
```

| Field                  | Type          | Required | Notes                                  |
|------------------------|---------------|----------|------------------------------------------|
| documents               | list          | yes (may be empty) | see `RetrievedDocument` below |
| documents[].text        | string        | yes      | the retrieved chunk content              |
| documents[].source      | string        | yes      | citeable origin (file, URL, doc id, ...) |
| documents[].score       | float 0.0-1.0 | yes      | relevance score                          |
| documents[].metadata    | object        | yes      | free-form (page, section, chunk id, ...) |
| citations                | list[string] | yes (may be empty) | typically `documents[].source` |
| confidence               | float or null | no       | overall retrieval confidence             |

## Error handling

Raise a `ContextRetrieverError` (or a subclass) on failure. The orchestrator
treats retrieval failures as non-fatal: it logs the failure and proceeds
without context rather than failing the whole request.

## Edge cases an implementation must handle without raising

- `top_k=0` → return an empty `ContextResult` (`documents=[]`).
- No matching documents → return an empty `ContextResult`, not an error.

## Reference implementation

See `app/context/mock_retriever.py` (`MockContextRetriever`) for a minimal
conforming implementation, and
`tests/test_context_retriever_contract.py` for the acceptance test suite.

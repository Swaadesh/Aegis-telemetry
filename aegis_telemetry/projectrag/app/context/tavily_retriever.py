import httpx

from app.context.base import (
    ContextResult,
    ContextRetriever,
    ContextRetrieverError,
    RetrievedDocument,
)

_SEARCH_URL = "https://api.tavily.com/search"


class TavilyContextRetriever(ContextRetriever):
    """Opt-in web search fallback, used only after a user confirms via
    POST /api/v1/websearch-confirm — never called on the first-pass
    /generate request. See docs/superpowers/specs/2026-08-09-web-search-
    fallback-design.md for why this exists despite Phase 1's no-external-
    tools boundary.
    """

    def __init__(
        self,
        api_key: str,
        max_results: int = 5,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._max_results = max_results
        self._client = client or httpx.AsyncClient(timeout=30.0)

    async def retrieve(self, query: str, top_k: int = 5) -> ContextResult:
        try:
            response = await self._client.post(
                _SEARCH_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"query": query, "max_results": top_k or self._max_results},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ContextRetrieverError(f"Tavily search failed: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ContextRetrieverError(f"Tavily search failed: {exc}") from exc

        payload = response.json()
        results = payload.get("results", [])[:top_k] if top_k else []

        documents = [
            RetrievedDocument(
                text=result["content"],
                source=result["url"],
                score=float(result["score"]),
                metadata={"title": result.get("title", "")},
            )
            for result in results
        ]
        return ContextResult(
            documents=documents,
            citations=[doc.source for doc in documents],
            confidence=documents[0].score if documents else None,
        )

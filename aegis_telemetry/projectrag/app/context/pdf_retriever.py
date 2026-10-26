import re
from pathlib import Path

from pypdf import PdfReader

from app.context.base import (
    ContextResult,
    ContextRetriever,
    ContextRetrieverError,
    RetrievedDocument,
)

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")

# Generic words dropped before scoring — without this, a query like "explain
# fastapi with a simple example" matched arbitrary chunks purely on "with"/
# "example"/"simple", so unrelated queries always looked like a document hit.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "in", "on", "at", "to",
    "for", "with", "without", "about", "is", "are", "was", "were", "be",
    "been", "being", "this", "that", "these", "those", "it", "its", "as",
    "by", "from", "into", "up", "down", "out", "so", "than", "then", "not",
    "do", "does", "did", "can", "could", "will", "would", "should", "may",
    "might", "explain", "example", "examples", "simple", "how", "what",
    "why", "when", "where", "who", "which",
}


def _tokenize(text: str) -> set[str]:
    words = {w.lower() for w in _WORD_RE.findall(text)}
    return words - _STOPWORDS


class PDFContextRetriever(ContextRetriever):
    """DEMO-ONLY retriever: extracts a single PDF into memory and ranks
    chunks by keyword overlap with the query.

    This is NOT the real RAG module (chunking strategy, embeddings, vector
    DB, ranking) that Atif Shamim owns per phase2.txt Step 12 — it exists
    only so the team can demo the pipeline answering real questions about
    TokenLens_Telemetry_and_SpendGuard.pdf from the terminal, without
    waiting on the real retrieval backend. Swap back to
    MockContextRetriever in app/dependencies.py when the demo is done, or
    forward to Atif's real ContextRetriever once it passes
    tests/test_context_retriever_contract.py.
    """

    def __init__(self, pdf_path: str, min_score: float = 0.3) -> None:
        path = Path(pdf_path)
        if not path.is_file():
            raise ContextRetrieverError(f"Knowledge base PDF not found: {path}")

        self._source_name = path.name
        self._min_score = min_score
        self._chunks: list[tuple[str, int]] = []  # (text, page_number)

        reader = PdfReader(str(path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            self._chunks.extend((paragraph, page_number) for paragraph in self._split_paragraphs(text))

    @staticmethod
    def _split_paragraphs(text: str, min_length: int = 40) -> list[str]:
        raw = [p.strip() for p in text.split("\n\n")]
        paragraphs = [p for p in raw if len(p) >= min_length]
        if paragraphs:
            return paragraphs
        return [text.strip()] if text.strip() else []

    async def retrieve(self, query: str, top_k: int = 5) -> ContextResult:
        if top_k <= 0 or not self._chunks:
            return ContextResult(documents=[], citations=[], confidence=None)

        query_tokens = _tokenize(query)
        if not query_tokens:
            return ContextResult(documents=[], citations=[], confidence=None)

        scored: list[tuple[float, str, int]] = []
        for text, page_number in self._chunks:
            chunk_tokens = _tokenize(text)
            overlap = len(query_tokens & chunk_tokens)
            if overlap == 0:
                continue
            score = overlap / len(query_tokens)
            if score < self._min_score:
                continue
            scored.append((score, text, page_number))

        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[:top_k]

        documents = [
            RetrievedDocument(
                text=text,
                source=f"{self._source_name}#page-{page_number}",
                score=score,
                metadata={"page": page_number},
            )
            for score, text, page_number in top
        ]
        return ContextResult(
            documents=documents,
            citations=[doc.source for doc in documents],
            confidence=documents[0].score if documents else None,
        )

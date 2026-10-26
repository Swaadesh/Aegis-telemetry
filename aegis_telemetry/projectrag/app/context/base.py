from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RetrievedDocument:
    """A single retrieved chunk, provider-agnostic (Qdrant/Milvus/Pinecone/...)."""

    text: str
    source: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class ContextResult:
    """Aggregate result of a retrieval call, ready to fold into a prompt."""

    documents: list[RetrievedDocument] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    confidence: float | None = None


class ContextRetrieverError(Exception):
    """Base class for retrieval-layer failures (mirrors ProviderError)."""


class ContextRetriever(ABC):
    """Common interface every retrieval backend (mock, Qdrant, Milvus, ...) implements.

    The orchestrator only ever talks to this interface, so the real RAG
    module (chunking, embeddings, ranking) can be swapped in later without
    touching orchestration, prompt building, or the API contract. See
    CONTRACT.md for the versioned input/output shape this interface exposes.
    """

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> ContextResult:
        """Return relevant context for `query`, up to `top_k` documents.

        Raises a ContextRetrieverError subclass on failure.
        """
        raise NotImplementedError

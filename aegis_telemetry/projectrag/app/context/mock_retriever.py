from app.context.base import ContextResult, ContextRetriever, RetrievedDocument


class MockContextRetriever(ContextRetriever):
    """Hardcoded stand-in for Atif's real RAG module.

    Lets the orchestrator, prompt builder, and API be built and tested end
    to end before a real vector store exists. Swapping this out for the
    real implementation should touch only app/dependencies.py.
    """

    async def retrieve(self, query: str, top_k: int = 5) -> ContextResult:
        documents = [
            RetrievedDocument(
                text="Project Aegis routes every AI request through a single governed gateway.",
                source="aegis-master-spec.pdf#overview",
                score=0.93,
                metadata={"section": "overview"},
            ),
            RetrievedDocument(
                text="Phase 1 implements a single-provider Gemini pipeline with no RAG or routing.",
                source="RoadPhase.txt#phase-1",
                score=0.87,
                metadata={"section": "phase-1"},
            ),
            RetrievedDocument(
                text="Phase 2 adds a retrieval interface and model router behind the orchestrator.",
                source="phase2.txt#goal",
                score=0.81,
                metadata={"section": "phase-2-goal"},
            ),
        ][:top_k]

        return ContextResult(
            documents=documents,
            citations=[doc.source for doc in documents],
            confidence=documents[0].score if documents else None,
        )

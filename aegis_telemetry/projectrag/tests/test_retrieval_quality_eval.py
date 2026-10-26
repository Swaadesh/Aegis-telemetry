"""Retrieval quality evaluation harness.

Runs against MockContextRetriever today (trivially passes, since the mock
always returns the same fixed documents). The point is having this harness
in place BEFORE Atif's real retriever exists — the day it's plugged in,
this becomes an immediate quality-regression signal instead of a confused
user filing a bug.
"""

import pytest

from app.context.mock_retriever import MockContextRetriever

# (query, expected_keyword_or_source substring that a good result should contain)
EVAL_CASES = [
    ("what is project aegis?", "aegis-master-spec.pdf"),
    ("what does phase 1 cover?", "RoadPhase.txt"),
    ("what is being built in phase 2?", "phase2.txt"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("query,expected_source_substring", EVAL_CASES)
async def test_retrieval_matches_expected_source(query: str, expected_source_substring: str) -> None:
    retriever = MockContextRetriever()

    result = await retriever.retrieve(query, top_k=5)

    sources = [doc.source for doc in result.documents]
    assert any(expected_source_substring in source for source in sources), (
        f"query={query!r} expected a document sourced from "
        f"{expected_source_substring!r}, got sources={sources}"
    )

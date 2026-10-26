from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized AI orchestration configuration.

    Everything the orchestrator/provider layer needs is read from here —
    never hardcode API keys or model parameters elsewhere.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    gemini_api_key: str
    model_name: str = "gemini-3.6-flash"
    # Gemini 3.x deprecates temperature/top_p/top_k sampling params in favor
    # of thinking_level. One of: minimal, low, medium, high.
    thinking_level: str = "medium"
    max_output_tokens: int = 2048
    timeout_seconds: float = 30.0
    max_retries: int = 2

    # Gemini safety category thresholds; see providers/gemini for usage.
    safety_block_threshold: str = "BLOCK_MEDIUM_AND_ABOVE"

    # Demo-only: path to the PDF backing PDFContextRetriever (see
    # app/context/pdf_retriever.py). Not part of the real RAG module —
    # that's Atif Shamim's Phase 2 handoff work per phase2.txt.
    knowledge_base_pdf_path: str = "../TokenLens_Telemetry_and_SpendGuard.pdf"
    # Chunks below this keyword-overlap score are treated as noise, not a
    # real match — without this, a handful of shared stopwords ("with",
    # "about") made PDFContextRetriever "find" the PDF for any query,
    # skipping the out-of-scope/web-search-confirmation path entirely.
    retrieval_min_score: float = 0.3

    # Demo-only guardrail: responses over this total token count are held
    # back and require the admin key (POST /api/v1/unlock) to release.
    # Not real authentication — see app/services/response_lock_store.py.
    response_token_limit: int = 1000
    admin_unlock_key: str = "7894"
    unlock_max_attempts: int = 2

    # Web search fallback (demo-grade, opt-in only — see
    # app/context/tavily_retriever.py and app/services/pending_search_store.py).
    tavily_api_key: str
    tavily_max_results: int = 5
    pending_search_ttl_seconds: int = 300


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # fields are populated from the environment/.env

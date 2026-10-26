# Backend — AI Orchestration Service

This is the AI orchestration layer for Project Aegis (see root `../CLAUDE.md`
for overall project context). Owner: Narendra Kumar, AI Development Lead.

## Stack

- Python 3.12, managed with `uv` (`pyproject.toml` / `uv.lock`)
- FastAPI + Uvicorn (ASGI web layer)
- `google-genai` — unified Google GenAI SDK for Gemini (do not use the
  deprecated `google-generativeai` package)
- Pydantic v2 + `pydantic-settings` for schemas and config
- pytest / pytest-asyncio for tests, ruff for lint, mypy for types, black
  for formatting

## Commands

```bash
uv sync                              # install/update dependencies
uv run uvicorn app.main:app --reload # run the dev server
uv run pytest                        # run tests
uv run ruff check .                  # lint
uv run mypy .                        # type check
```

## Architecture rule (non-negotiable)

The API layer never talks to a model provider directly. Every request goes
through the **AI Orchestrator**:

```text
API route -> AI Orchestrator -> Prompt Builder -> Provider -> Response Formatter -> API route
```

This keeps the gateway-facing contract identical as more providers, routing,
and RAG are added later. When adding a new capability, ask "does this belong
in the orchestrator, or in a provider/service it calls?" — never bolt logic
directly onto the API route.

## Package layout (implemented)

```text
app/
├── main.py                          FastAPI app, /health, router registration
├── dependencies.py                  wires Settings -> GeminiProvider -> AIOrchestrator (cached singleton)
├── api/
│   └── routes.py                    POST /api/v1/generate, /api/v1/unlock, GET /api/v1/health — thin, no business logic
├── orchestrator/
│   └── orchestrator.py              AIOrchestrator — the only entry point providers are called from;
│                                     retries on timeout, raises AIOrchestrationError on failure,
│                                     raises ResponseLockedError when response_token_limit is crossed
├── providers/
│   ├── base.py                      AIProvider ABC + ProviderResponse + ProviderError hierarchy
│   └── gemini/gemini_provider.py    wraps google-genai's async client for model gemini-3.6-flash
├── prompts/prompt_builder.py        system/user prompt templates, separate from business logic
├── models/schemas.py                AIRequest / AIResponse / ErrorResponse / TokenUsage (pydantic)
├── config/settings.py               pydantic-settings Settings — all config (key, model, thinking level,
│                                     timeout, max tokens, safety settings, response_token_limit,
│                                     admin_unlock_key) centralized here, never hardcoded
├── services/
│   ├── response_formatter.py        validates + normalizes provider output into the standard envelope
│   └── response_lock_store.py       ResponseLockStore — in-memory-only holding area for responses
│                                     gated behind the admin unlock key (demo-grade, not real auth)
└── utils/logging.py                 structured per-request logging (log_request: request_id, latency,
                                      tokens, status, threshold) + log_lock_resolution (unlock outcomes:
                                      unlocked / attempt_failed / blocked)
tests/
├── test_orchestrator.py             retry/no-retry logic, concurrency, success/failure paths (stub provider)
├── test_gemini_provider.py          Gemini SDK call + error translation (mocked client)
├── test_api.py                      normal/empty/long prompt, timeout -> 502, concurrent requests, /health
└── test_response_lock.py            lock/unlock flow end-to-end, plus the structured log lines emitted
                                      for lock creation (threshold) and unlock resolution (event)
aegis_cli.py                          interactive terminal demo client (not part of app/) — hits
                                       /api/v1/generate and /api/v1/unlock like any other gateway consumer
```

Status: fully implemented and verified — 43/43 tests passing, `ruff check .`
and `mypy .` both clean, and manually smoke-tested against the live Gemini
API (`gemini-3.6-flash`) for both a successful generation and a structured
error response (invalid key -> `502` with `error_type`/`message`).

## Config & secrets

- All configuration (API keys, model name, thinking level, timeouts, token
  limits, safety settings) lives in `app/config/settings.py` via
  `pydantic-settings`, backed by a `.env` file.
- Never hardcode API keys or model parameters in provider/orchestrator code.
- `.env` must stay out of version control (already covered by `.gitignore`);
  commit a `.env.example` with placeholder values instead.
- Gemini 3.x deprecates the old sampling params (`temperature`, `top_p`,
  `top_k`) — currently ignored, will error in future model generations.
  Use `thinking_level` (`minimal`/`low`/`medium`/`high`) instead, via
  `google.genai.types.ThinkingConfig`. Don't reintroduce temperature-style
  config when touching `gemini_provider.py`.

## Current phase scope

Phase 1 = Gemini only, no RAG, no routing, no agents/tools/memory/MCP. See
root `RoadPhase.txt` and `../summary.txt` for the full Phase 1 checklist
(1A–1J) and implementation order. Don't add capabilities from later phases
without being asked.

## Testing expectations

Cover, at minimum, per `RoadPhase.txt` Phase 1J: a normal prompt, an empty
prompt, a long prompt, an invalid API key, a provider timeout, and concurrent
requests. Mock the `google-genai` client in tests — tests must not require a
live API key or make real network calls.

## Known gotchas

- `PDFContextRetriever` (`app/context/pdf_retriever.py`) scores chunks by
  keyword overlap. It filters out `_STOPWORDS` (generic words like
  "with"/"example"/"explain") and drops anything below
  `Settings.retrieval_min_score` (default 0.3) before ranking — without
  this, overlap on common words made any query "match" the PDF, silently
  skipping the out-of-scope/web-search path. Keep both when touching this
  file.
- `aegis_cli.py`'s regular prompts (`Confirm.ask`/`Prompt.ask` for user id,
  role, query, y/n) accept piped stdin fine, e.g.
  `printf 'n\nn\nquery text\nn\nexit\n' | uv run python aegis_cli.py`. The
  admin-key prompt (`_prompt_masked`) does not — it reads raw console input
  (`msvcrt`/`termios`) and hangs on piped stdin. To exercise the
  lock/unlock flow non-interactively, call the API directly instead:
  `POST /api/v1/generate` then `POST /api/v1/unlock` with `admin_key`
  (default `7894` unless overridden in `.env`). The server must already be
  running (`uv run uvicorn app.main:app --reload`); check with
  `curl http://127.0.0.1:8000/api/v1/health`.

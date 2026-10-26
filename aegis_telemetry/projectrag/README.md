# Project Aegis — AI Orchestration Service

The AI orchestration layer of **Project Aegis**, an Enterprise AI Gateway:
employees never call an LLM directly — every request flows through Aegis
(auth -> DLP -> AI orchestration -> LLM provider -> response) so the
organization gets a single, governed, auditable entry point for AI usage.

This repo owns the **AI orchestration layer only** — prompt building, model
orchestration, provider integration (Gemini first), response formatting,
logging/observability, and error handling for the AI pipeline. It does not
own the frontend dashboard, DLP rules, OAuth/gateway auth, gateway
networking, vector database security, telemetry dashboards, or QA testing.

## Status

**Phase 1 — AI Orchestration Foundation: built and verified.** Single
provider (Gemini, `gemini-3.6-flash`), no multi-model routing, no
agents/tools/memory. The full pipeline
`Client -> FastAPI -> AI Orchestrator -> Gemini Provider -> Response` is
implemented with config, prompt management, logging, retry/error handling,
and tests — smoke-tested against the live Gemini API.

## Stack

- Python 3.12, managed with [`uv`](https://github.com/astral-sh/uv)
- FastAPI + Uvicorn
- `google-genai` (unified Google GenAI SDK)
- Pydantic v2 + `pydantic-settings`
- pytest / pytest-asyncio, ruff, mypy, black

## Architecture

The API layer never talks to a model provider directly — every request goes
through the AI Orchestrator:

```text
API route -> AI Orchestrator -> Prompt Builder -> Provider -> Response Formatter -> API route
```

```text
app/
├── main.py                          FastAPI app, /health, router registration
├── dependencies.py                  wires Settings -> GeminiProvider -> AIOrchestrator
├── api/routes.py                    POST /api/v1/generate, /api/v1/unlock, GET /api/v1/health
├── orchestrator/orchestrator.py     AIOrchestrator — retry, error handling, response lock
├── providers/
│   ├── base.py                      AIProvider interface (for future providers)
│   └── gemini/gemini_provider.py    wraps google-genai for gemini-3.6-flash
├── prompts/prompt_builder.py        system/user prompt construction
├── models/schemas.py                AIRequest / AIResponse / ErrorResponse / TokenUsage
├── config/settings.py               centralized configuration (pydantic-settings)
├── services/
│   ├── response_formatter.py        normalizes provider output into a standard envelope
│   └── response_lock_store.py       demo-grade response-lock / admin-unlock guardrail
└── utils/logging.py                 structured per-request logging
```

## Getting started

```bash
uv sync                              # install dependencies
cp .env.example .env                 # add your GEMINI_API_KEY, etc.
uv run uvicorn app.main:app --reload # run the dev server
```

```bash
uv run pytest                        # run tests
uv run ruff check .                  # lint
uv run mypy .                        # type check
```

## Scope

Deferred to later phases (not built here): RAG, vector databases,
multi-model routing, LangGraph, multi-agent orchestration, function/tool
calling, MCP. See `CLAUDE.md` for full project conventions and phase
boundaries.

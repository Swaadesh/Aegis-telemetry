from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from app.api.routes import router
from app.config.settings import get_settings

console = Console()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    console.print(
        Panel.fit(
            "[bold cyan]Welcome to PROJECT AEGIS[/bold cyan]\n"
            "[dim]Enterprise AI Gateway - Orchestration Layer[/dim]\n"
            f"Model: [bold green]{settings.model_name}[/bold green]",
            border_style="cyan",
        )
    )
    yield


app = FastAPI(
    title="Project Aegis — AI Orchestration Service",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(router)


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"status": "error", "error_type": "invalid_request", "message": str(exc)},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

"""Interactive terminal demo client for Project Aegis.

Not part of the orchestration service (app/) — this is a thin HTTP client
for demoing the gateway to the team from a terminal, with no frontend
involved. It talks to the running Aegis server the same way any other
client/gateway consumer would: only through POST /api/v1/generate, never
to a model provider directly.

Usage:
    uv run python aegis_cli.py [--url http://127.0.0.1:8000]

Requires the Aegis server to already be running in another terminal:
    uv run uvicorn app.main:app --reload
"""

import argparse
import os
import sys
from datetime import datetime

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

console = Console()

DEFAULT_URL = os.environ.get("AEGIS_API_URL", "http://127.0.0.1:8000")


def _prompt_masked(label: str) -> str:
    """Read a line of input echoing '*' per character instead of the raw
    text or nothing at all — so the admin key is visibly masked rather than
    silently hidden. Falls back to fully-hidden input when stdin isn't a
    real terminal (e.g. piped input in tests)."""
    console.print(f"{label}: ", end="")
    if not sys.stdin.isatty():
        return Prompt.ask("", password=True)

    chars: list[str] = []
    try:
        if sys.platform == "win32":
            import msvcrt

            while True:
                ch = msvcrt.getwch()
                if ch in ("\r", "\n"):
                    break
                if ch == "\x08":  # backspace
                    if chars:
                        chars.pop()
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                    continue
                if ch == "\x03":  # Ctrl+C
                    raise KeyboardInterrupt
                chars.append(ch)
                sys.stdout.write("*")
                sys.stdout.flush()
        else:
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                while True:
                    ch = sys.stdin.read(1)
                    if ch in ("\r", "\n"):
                        break
                    if ch in ("\x7f", "\x08"):  # backspace
                        if chars:
                            chars.pop()
                            sys.stdout.write("\b \b")
                            sys.stdout.flush()
                        continue
                    if ch == "\x03":
                        raise KeyboardInterrupt
                    chars.append(ch)
                    sys.stdout.write("*")
                    sys.stdout.flush()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    finally:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return "".join(chars)


def print_banner() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]Welcome to PROJECT AEGIS[/bold cyan]\n"
            "[dim]Enterprise AI Gateway - Terminal Demo Client[/dim]",
            border_style="cyan",
        )
    )


def print_success(body: dict, received_at: datetime) -> None:
    is_web = body.get("source") == "web"
    title = "AEGIS Response (Web Search)" if is_web else "AEGIS Response"
    border = "yellow" if is_web else "green"
    console.print(Panel(body["text"], title=title, border_style=border))
    if is_web:
        console.print("[dim]Source: web search results, not the knowledge base document.[/dim]")

    token_usage = body.get("token_usage", {})
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("[bold]Received At[/bold]", received_at.isoformat(timespec="seconds"))
    table.add_row("[bold]Request ID[/bold]", body["request_id"])
    table.add_row("[bold]Model[/bold]", body["model"])
    table.add_row("[bold]Status[/bold]", f"[green]{body['status']}[/green]")
    table.add_row("[bold]Source[/bold]", body.get("source", "document"))
    table.add_row("[bold]Latency (ms)[/bold]", f"{body['latency_ms']:.1f}")
    table.add_row("[bold]Prompt Tokens[/bold]", str(token_usage.get("prompt_tokens")))
    table.add_row("[bold]Completion Tokens[/bold]", str(token_usage.get("completion_tokens")))
    table.add_row("[bold]Total Tokens[/bold]", str(token_usage.get("total_tokens")))
    if is_web:
        table.add_row("[bold]Web Results Used[/bold]", str(body.get("documents_used", 0)))
    else:
        table.add_row("[bold]Documents Used[/bold]", f"{body.get('documents_used', 0)} pages")
    table.add_row("[bold]Citations[/bold]", ", ".join(body.get("citations", [])) or "-")
    console.print(table)


def print_error(status_code: int, body: dict) -> None:
    console.print(
        Panel(
            f"[bold]{body.get('error_type', 'unknown_error')}[/bold]\n{body.get('message', '')}",
            title=f"AEGIS Error ({status_code})",
            border_style="red",
        )
    )


def print_locked(body: dict) -> None:
    is_web = body.get("source") == "web"
    title = "AEGIS Response Locked (Web Search)" if is_web else "AEGIS Response Locked"
    console.print(
        Panel(
            f"{body.get('message', '')}\n"
            f"[dim]Request ID: {body.get('request_id', '')} — "
            f"{body.get('attempts_remaining', 0)} attempt(s) remaining[/dim]",
            title=title,
            border_style="yellow",
        )
    )


def print_out_of_scope(body: dict) -> None:
    console.print(
        Panel(
            "No answer found in the knowledge base.",
            title="AEGIS — Out of Scope",
            border_style="cyan",
        )
    )


def try_web_search(
    client: httpx.Client, request_id: str, query: str, headers: dict[str, str]
) -> dict | None:
    """Ask the user whether to fall back to a web search; POST the decision
    to /api/v1/websearch-confirm either way, so the server always resolves
    the pending entry (declining still evicts it and logs the outcome)."""
    confirm = Confirm.ask(
        "[bold cyan]Nothing found in the knowledge base. Search the web instead?[/bold cyan]",
        default=True,
    )
    if not confirm:
        resp = client.post(
            "/api/v1/websearch-confirm",
            json={"request_id": request_id, "confirm": confirm},
            headers=headers,
        )
        body = resp.json()
    else:
        with console.status(f"[bold cyan]Searching web: {query}[/bold cyan]"):
            resp = client.post(
                "/api/v1/websearch-confirm",
                json={"request_id": request_id, "confirm": confirm},
                headers=headers,
            )
        body = resp.json()
    if resp.status_code != 200:
        console.print(
            Panel(
                body.get("message", ""),
                title=f"Web Search Failed ({resp.status_code})",
                border_style="red",
            )
        )
        return None
    if body.get("status") == "declined":
        console.print(Panel(body.get("message", "OK."), title="AEGIS", border_style="dim"))
        return None
    return body


def try_unlock(client: httpx.Client, request_id: str, headers: dict[str, str]) -> dict | None:
    """Prompt for the admin key and call /api/v1/unlock until it succeeds,
    the user gives up, or the server reports the attempts are exhausted."""
    while True:
        if not Confirm.ask("[bold yellow]Enter admin secret key to unlock?[/bold yellow]", default=True):
            return None
        admin_key = _prompt_masked("[bold yellow]Admin key[/bold yellow]")
        resp = client.post(
            "/api/v1/unlock",
            json={"request_id": request_id, "admin_key": admin_key},
            headers=headers,
        )
        body = resp.json()
        if resp.status_code == 200:
            return body
        console.print(
            Panel(body.get("message", ""), title=f"Unlock Failed ({resp.status_code})", border_style="red")
        )
        if resp.status_code != 401:
            return None  # 403 exhausted / 404 expired — no point retrying


def run(base_url: str) -> None:
    print_banner()
    headers = {}
    if Confirm.ask("[dim]Enter a demo user id?[/dim]", default=False):
        user_id = Prompt.ask("[dim]User id[/dim]")
        if user_id:
            headers["X-User-Id"] = user_id
    if Confirm.ask("[dim]Enter a demo role?[/dim]", default=False):
        role = Prompt.ask("[dim]Role[/dim]")
        if role:
            headers["X-User-Role"] = role

    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        while True:
            query = Prompt.ask("\n[bold]Query[/bold]")
            if query.strip().lower() in {"exit", "quit"}:
                console.print("[dim]Goodbye.[/dim]")
                break
            if not query.strip():
                continue

            try:
                with console.status(
                    "[bold cyan]AEGIS is generating a response using gemini-3.6-flash...[/bold cyan]"
                ):
                    response = client.post(
                        "/api/v1/generate", json={"prompt": query}, headers=headers
                    )
            except httpx.ConnectError:
                console.print(
                    Panel(
                        "Could not reach the Aegis server. Is it running?\n"
                        "[bold]uv run uvicorn app.main:app --reload[/bold]",
                        title="Connection Error",
                        border_style="red",
                    )
                )
                continue

            received_at = datetime.now().astimezone()
            body = response.json()
            if response.status_code == 200 and body.get("status") == "locked":
                print_locked(body)
                unlocked = try_unlock(client, body["request_id"], headers)
                if unlocked is not None:
                    print_success(unlocked, datetime.now().astimezone())
            elif response.status_code == 200 and body.get("status") == "out_of_scope":
                print_out_of_scope(body)
                web_result = try_web_search(client, body["request_id"], query, headers)
                if web_result is not None:
                    if web_result.get("status") == "locked":
                        print_locked(web_result)
                        unlocked = try_unlock(client, web_result["request_id"], headers)
                        if unlocked is not None:
                            print_success(unlocked, datetime.now().astimezone())
                    else:
                        print_success(web_result, datetime.now().astimezone())
            elif response.status_code == 200:
                print_success(body, received_at)
            else:
                print_error(response.status_code, body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Project Aegis terminal demo client")
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL of the Aegis server")
    args = parser.parse_args()
    run(args.url)


if __name__ == "__main__":
    main()

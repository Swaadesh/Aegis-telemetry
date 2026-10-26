"""
Project Aegis - Telemetry Storage
Defines tables and read/write functions for the telemetry dashboard.

Schema here matches Narendra's app/utils/logging.py EXACTLY (field names,
types, nullability) — that file is the source of truth. Three log shapes,
three tables:

  1. log_request()             -> execution_logs        (one row per AI request)
  2. log_lock_resolution()     -> lock_events            (Spend Guard unlock flow)
  3. log_websearch_resolution()-> websearch_events        (out-of-scope -> web search flow)

All three share request_id, so any request's full story -- generate,
lock, unlock attempts, web search confirmation -- can be reconstructed
by joining on it.
"""

import sqlite3
import json
from contextlib import contextmanager

DB_PATH = "telemetry.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS execution_logs (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp              TEXT NOT NULL,
    request_id             TEXT NOT NULL,
    user_id                TEXT,
    role                   TEXT,
    model                  TEXT,
    prompt_tokens          INTEGER,
    completion_tokens      INTEGER,
    total_tokens           INTEGER,
    latency_ms             REAL,
    status                 TEXT,
    error_type             TEXT,
    retrieval_latency_ms   REAL,
    documents_retrieved    INTEGER,
    chunk_count            INTEGER,
    citation_ids           TEXT,
    search_score           REAL,
    threshold               INTEGER
);

CREATE TABLE IF NOT EXISTS lock_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT NOT NULL,
    request_id          TEXT NOT NULL,
    event                TEXT NOT NULL,
    user_id              TEXT,
    role                  TEXT,
    attempts_remaining    INTEGER
);

CREATE TABLE IF NOT EXISTS websearch_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT NOT NULL,
    request_id   TEXT NOT NULL,
    event         TEXT NOT NULL,
    user_id       TEXT,
    role           TEXT
);
"""


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def insert_execution_log(entry: dict):
    """Insert one row from a log_request() JSON line."""
    citation_ids = entry.get("citation_ids") or []
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO execution_logs
            (timestamp, request_id, user_id, role, model, prompt_tokens,
             completion_tokens, total_tokens, latency_ms, status, error_type,
             retrieval_latency_ms, documents_retrieved, chunk_count,
             citation_ids, search_score, threshold)
            VALUES (:timestamp, :request_id, :user_id, :role, :model,
                    :prompt_tokens, :completion_tokens, :total_tokens,
                    :latency_ms, :status, :error_type, :retrieval_latency_ms,
                    :documents_retrieved, :chunk_count, :citation_ids,
                    :search_score, :threshold)
            """,
            {**entry, "citation_ids": json.dumps(citation_ids)},
        )


def fetch_all_execution_logs():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM execution_logs ORDER BY timestamp DESC"
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["citation_ids"] = json.loads(d["citation_ids"] or "[]")
            results.append(d)
        return results


def execution_log_count():
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM execution_logs").fetchone()[0]


def insert_lock_event(entry: dict):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO lock_events
            (timestamp, request_id, event, user_id, role, attempts_remaining)
            VALUES (:timestamp, :request_id, :event, :user_id, :role, :attempts_remaining)
            """,
            entry,
        )


def fetch_all_lock_events():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM lock_events ORDER BY timestamp DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def insert_websearch_event(entry: dict):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO websearch_events (timestamp, request_id, event, user_id, role)
            VALUES (:timestamp, :request_id, :event, :user_id, :role)
            """,
            entry,
        )


def fetch_all_websearch_events():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM websearch_events ORDER BY timestamp DESC"
        ).fetchall()
        return [dict(row) for row in rows]

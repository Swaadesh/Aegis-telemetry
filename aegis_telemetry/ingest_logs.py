"""
Project Aegis - Live Log Ingester
Reads Narendra's structured JSON log lines from stdin and writes each one
into the right telemetry table, live, as they're printed.

His logging (app/utils/logging.py) only writes to stdout right now -- so
the way to get real data flowing into this dashboard without him changing
anything on his side is to PIPE his server's output into this script.

Usage (run from his repo's directory, or point --cwd at it):
    uv run uvicorn app.main:app --reload | python ingest_logs.py

Or, to also see the server output live in your own terminal while it's
being captured:
    uv run uvicorn app.main:app --reload | tee /dev/stderr | python ingest_logs.py

Each line is one JSON object. This script figures out which table it
belongs to based on which fields are present:
    - has "status"                  -> execution_logs   (log_request)
    - has "event" + "attempts_remaining" key -> lock_events (log_lock_resolution)
    - has "event", no "attempts_remaining" key -> websearch_events (log_websearch_resolution)

Non-JSON lines (uvicorn's own "INFO: ..." startup/access logs) are just
printed through untouched and skipped -- this script only acts on the
structured JSON lines his logger emits.
"""

import sys
import json

from storage import (
    init_db,
    insert_execution_log,
    insert_lock_event,
    insert_websearch_event,
)


def classify_and_insert(entry: dict) -> str:
    """Insert one parsed log line into the right table. Returns a label
    for what was inserted, used for the console feedback line."""
    if "status" in entry:
        insert_execution_log(entry)
        return f"execution_log  status={entry.get('status')}  request_id={entry.get('request_id')}"

    if "event" in entry:
        if "attempts_remaining" in entry:
            insert_lock_event(entry)
            return f"lock_event     event={entry.get('event')}  request_id={entry.get('request_id')}"
        else:
            insert_websearch_event(entry)
            return f"websearch_event event={entry.get('event')}  request_id={entry.get('request_id')}"

    return None


def main():
    init_db()
    print("[ingest_logs] Listening for JSON log lines on stdin... "
          "(pipe the uvicorn server output into this script)", file=sys.stderr)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            # Not a JSON log line (e.g. uvicorn's "INFO: ..." lines) -- ignore.
            continue

        label = classify_and_insert(entry)
        if label:
            print(f"[ingest_logs] stored -> {label}", file=sys.stderr)


if __name__ == "__main__":
    main()

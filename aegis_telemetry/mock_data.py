"""
Project Aegis - Mock Telemetry Data Generator
Generates fake log entries matching Narendra's REAL schema (see
app/utils/logging.py in his repo) so the dashboard can be built and
demoed before/without a live pipe from his server.

Once ingest_logs.py is running against his real stdout, stop running this
and the dashboard will show real data instead -- nothing else changes.

Usage:
    python mock_data.py            # generates ~150 sample requests
    python mock_data.py 300        # custom count
"""

import sys
import random
import uuid
from datetime import datetime, timedelta, timezone

from storage import init_db, insert_execution_log, insert_lock_event, insert_websearch_event

USERS = [
    ("101", "engineering"), ("102", "engineering"),
    ("103", "product"), ("104", "product"),
    ("105", "sales"), ("106", "legal"),
    ("107", "data_analytics"), ("108", "cybersecurity"),
]
MODEL = "gemini-3.6-flash"
CITATION_POOL = [
    "https://medium.com/@patel_shrijit/fastapi-concepts-explained",
    "https://refine.dev/blog/introduction-to-fast-api",
    "https://www.geeksforgeeks.org/python/introduction-to-fastapi",
    "https://www.datacamp.com/tutorial/introduction-fastapi-tutorial",
]
THRESHOLD = 1000  # matches the real value seen in Narendra's demo


def random_ts(days_back=30):
    return (datetime.now(timezone.utc)
            - timedelta(days=random.uniform(0, days_back), hours=random.uniform(0, 24))).isoformat()


def generate_requests(n: int):
    """
    For each simulated request, generate the same event sequence his
    server actually produces: an execution_log row, and -- depending on
    outcome -- either lock_events (Spend Guard) or websearch_events
    (out-of-scope -> web search) rows tied by request_id.
    """
    for _ in range(n):
        user_id, role = random.choice(USERS)
        request_id = str(uuid.uuid4())
        ts = random_ts()

        roll = random.random()

        if roll < 0.15:
            # out_of_scope: nothing found in knowledge base
            insert_execution_log({
                "timestamp": ts, "request_id": request_id, "user_id": user_id, "role": role,
                "model": MODEL, "prompt_tokens": None, "completion_tokens": None,
                "total_tokens": None, "latency_ms": round(random.uniform(1, 5), 1),
                "status": "out_of_scope", "error_type": None,
                "retrieval_latency_ms": round(random.uniform(1, 5), 1),
                "documents_retrieved": 0, "chunk_count": None,
                "citation_ids": [], "search_score": None, "threshold": None,
            })
            # user may or may not confirm falling back to web search
            event = random.choice(["confirmed", "confirmed", "declined"])
            insert_websearch_event({
                "timestamp": ts, "request_id": request_id, "event": event,
                "user_id": user_id, "role": role,
            })

        elif roll < 0.30:
            # locked: crossed the spend/token threshold
            prompt_t, completion_t = random.randint(400, 1800), random.randint(100, 700)
            insert_execution_log({
                "timestamp": ts, "request_id": request_id, "user_id": user_id, "role": role,
                "model": MODEL, "prompt_tokens": prompt_t, "completion_tokens": completion_t,
                "total_tokens": prompt_t + completion_t,
                "latency_ms": round(random.uniform(2000, 10000), 1),
                "status": "locked", "error_type": None,
                "retrieval_latency_ms": round(random.uniform(200, 2000), 1),
                "documents_retrieved": random.randint(1, 5), "chunk_count": random.randint(1, 5),
                "citation_ids": random.sample(CITATION_POOL, k=random.randint(1, len(CITATION_POOL))),
                "search_score": round(random.uniform(0.6, 0.95), 4), "threshold": THRESHOLD,
            })
            n_failed = random.randint(0, 2)
            for i in range(n_failed):
                insert_lock_event({
                    "timestamp": ts, "request_id": request_id, "event": "attempt_failed",
                    "user_id": user_id, "role": role, "attempts_remaining": 2 - i,
                })
            final_event = random.choice(["unlocked", "unlocked", "blocked"])
            insert_lock_event({
                "timestamp": ts, "request_id": request_id, "event": final_event,
                "user_id": user_id, "role": role,
                "attempts_remaining": None if final_event == "unlocked" else 0,
            })

        else:
            # normal success
            prompt_t, completion_t = random.randint(50, 800), random.randint(20, 500)
            insert_execution_log({
                "timestamp": ts, "request_id": request_id, "user_id": user_id, "role": role,
                "model": MODEL, "prompt_tokens": prompt_t, "completion_tokens": completion_t,
                "total_tokens": prompt_t + completion_t,
                "latency_ms": round(random.uniform(200, 4000), 1),
                "status": "success", "error_type": None,
                "retrieval_latency_ms": round(random.uniform(50, 800), 1),
                "documents_retrieved": random.randint(1, 5), "chunk_count": random.randint(1, 5),
                "citation_ids": random.sample(CITATION_POOL, k=random.randint(0, 2)),
                "search_score": round(random.uniform(0.5, 0.95), 4), "threshold": None,
            })


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    init_db()
    generate_requests(n)
    print(f"Inserted {n} simulated requests (with matching lock/websearch events).")

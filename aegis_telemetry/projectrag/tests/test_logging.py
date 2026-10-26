import json
import logging

from app.utils.logging import log_websearch_resolution


def test_log_websearch_resolution_emits_structured_line(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="aegis.ai"):
        log_websearch_resolution(request_id="r1", event="confirmed", user_id="u1", role="analyst")

    lines = [json.loads(r.message) for r in caplog.records if r.name == "aegis.ai"]
    assert len(lines) == 1
    assert lines[0]["request_id"] == "r1"
    assert lines[0]["event"] == "confirmed"
    assert lines[0]["user_id"] == "u1"
    assert lines[0]["role"] == "analyst"

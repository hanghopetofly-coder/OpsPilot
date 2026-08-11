"""Static contract checks for the AIOps SSE consumer."""

from __future__ import annotations

from pathlib import Path


def test_frontend_does_not_append_complete_response_after_report_event() -> None:
    source = Path("static/app.js").read_text(encoding="utf-8")
    method_start = source.index("async sendAIOpsRequest")
    declaration = source.index("let reportReceived = false;", method_start)
    first_assignment = source.index("reportReceived = true;", method_start)

    assert method_start < declaration < first_assignment
    assert source.count("let reportReceived = false;") == 1
    assert source.count("reportReceived = true;") == 2
    assert source.count("sseMessage.response && !reportReceived") == 2


def test_frontend_propagates_aiops_error_events_independent_of_message_text() -> None:
    source = Path("static/app.js").read_text(encoding="utf-8")
    method_start = source.index("async sendAIOpsRequest")
    method_end = source.index("\n    updateAIOpsStreamContent(", method_start)
    method_source = source[method_start:method_end]

    assert method_source.count("streamError.isAIOpsStreamError = true;") == 2
    assert method_source.count("if (e.isAIOpsStreamError) throw e;") == 2
    assert "e.message.includes('智能运维')" not in method_source

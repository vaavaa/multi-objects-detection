from __future__ import annotations

from app.agents import run_info_responder_answer


def test_run_info_responder_mock(monkeypatch, tmp_path):
    monkeypatch.setenv("INFO_AGENT_MOCK", "1")
    out = run_info_responder_answer("Какой у нас график?", chat_id="c-1")
    assert "MOCK" in out.answer
    assert "llama3.2" in out.answer or "MOCK" in out.notes

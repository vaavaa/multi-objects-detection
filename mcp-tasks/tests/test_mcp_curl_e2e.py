"""
Проверки «снаружи» через curl: GET /health и MCP JSON-RPC на POST /mcp.

Сервер в режиме Open WebUI (FASTMCP_JSON_RESPONSE=false, STATELESS=false) отвечает
на RPC как text/event-stream (SSE); Accept должен включать и json, и event-stream.
После initialize в заголовках приходит mcp-session-id — его нужно передавать дальше.

Запуск:

  MCP_TASKS_E2E_URL=http://127.0.0.1:8084 pytest -m e2e tests/test_mcp_curl_e2e.py -v
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

PROTOCOL_VERSION = "2025-03-26"


def _base_url() -> str | None:
    u = os.getenv("MCP_TASKS_E2E_URL", "").strip().rstrip("/")
    return u or None


@pytest.fixture(scope="module")
def base_url() -> str:
    u = _base_url()
    if not u:
        pytest.skip("Задайте MCP_TASKS_E2E_URL, например http://127.0.0.1:8084")
    return u


@pytest.fixture(scope="module")
def curl_exe() -> str:
    exe = shutil.which("curl")
    if not exe:
        pytest.skip("Нужен curl в PATH")
    return exe


def _response_headers_dict(header_text: str) -> dict[str, str]:
    d: dict[str, str] = {}
    for line in header_text.splitlines():
        line = line.strip("\r")
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        d[k.strip().lower()] = v.strip()
    return d


def _content_type(header_text: str) -> str:
    return _response_headers_dict(header_text).get("content-type", "").lower()


def _parse_mcp_body(status: int, header_text: str, raw: bytes) -> dict[str, Any]:
    if status == 202 or not raw.strip():
        return {}
    ct_sub = _content_type(header_text)
    text = raw.decode("utf-8")
    if "application/json" in ct_sub:
        return json.loads(text)
    if "text/event-stream" in ct_sub or text.startswith("event:") or "data:" in text:
        for line in text.splitlines():
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload:
                    return json.loads(payload)
        raise AssertionError(f"SSE без строки data: {text[:400]!r}")
    return json.loads(text)


def _curl(
    curl: str,
    tmp_path: Path,
    *,
    method: str,
    url: str,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, str, bytes]:
    hdr_path = tmp_path / "curl-headers.txt"
    body_path = tmp_path / "curl-body.bin"
    cmd: list[str] = [
        curl,
        "-sS",
        "-X",
        method,
        url,
        "-D",
        str(hdr_path),
        "-o",
        str(body_path),
        "-w",
        "%{http_code}",
    ]
    if data is not None:
        cmd.extend(["-d", data.decode("utf-8")])
    for k, v in (headers or {}).items():
        cmd.extend(["-H", f"{k}: {v}"])
    out = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if out.returncode != 0:
        raise AssertionError(f"curl failed rc={out.returncode} stderr={out.stderr!r}")
    status = int((out.stdout or "0").strip())
    first = hdr_path.read_bytes().split(b"\r\n\r\n", 1)[0]
    header_text = first.decode("utf-8", errors="replace")
    body = body_path.read_bytes()
    return status, header_text, body


def _mcp_post(
    curl: str,
    base_url: str,
    tmp_path: Path,
    payload: dict[str, Any],
    *,
    session_id: str | None = None,
    protocol_version: str | None = None,
    tmp_suffix: str = "x",
) -> tuple[int, dict[str, Any], str]:
    """POST JSON-RPC на /mcp; возвращает (status, тело как dict, сырой блок заголовков)."""
    sub = tmp_path / tmp_suffix
    sub.mkdir(parents=True, exist_ok=True)
    h = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    if session_id:
        h["mcp-session-id"] = session_id
    if protocol_version:
        h["mcp-protocol-version"] = protocol_version
    status, hdr, raw = _curl(
        curl,
        sub,
        method="POST",
        url=f"{base_url}/mcp",
        data=json.dumps(payload).encode("utf-8"),
        headers=h,
    )
    try:
        parsed = _parse_mcp_body(status, hdr, raw)
    except (json.JSONDecodeError, AssertionError) as e:
        raise AssertionError(
            f"MCP parse error status={status} ct={_content_type(hdr)!r} body={raw[:500]!r}"
        ) from e
    return status, parsed, hdr


def _mcp_bootstrap(curl: str, base: str, tmp: Path) -> tuple[str, str]:
    st, init, hdr = _mcp_post(
        curl,
        base,
        tmp,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "pytest-curl-e2e", "version": "1.0"},
            },
        },
        tmp_suffix="init",
    )
    assert st == 200, init
    assert init.get("result"), init
    ver = str(init["result"].get("protocolVersion", PROTOCOL_VERSION))
    sess = _response_headers_dict(hdr).get("mcp-session-id")
    assert sess, f"нет mcp-session-id в ответе initialize, заголовки: {hdr[:500]!r}"
    st2, _, _ = _mcp_post(
        curl,
        base,
        tmp,
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        session_id=sess,
        protocol_version=ver,
        tmp_suffix="notif",
    )
    assert st2 == 202
    return ver, sess


@pytest.mark.e2e
def test_health_get_json(curl_exe, base_url, tmp_path):
    url = f"{base_url}/health"
    code, hdr, body = _curl(curl_exe, tmp_path, method="GET", url=url)
    assert code == 200
    assert "application/json" in hdr.lower()
    data = json.loads(body.decode("utf-8"))
    assert data.get("status") == "ok"
    assert data.get("mcp_url_path") == "/mcp"
    assert "sqlite_path" in data


@pytest.mark.e2e
def test_mcp_initialize_and_tools_list(curl_exe, base_url, tmp_path):
    ver, sess = _mcp_bootstrap(curl_exe, base_url, tmp_path)
    st, msg, _ = _mcp_post(
        curl_exe,
        base_url,
        tmp_path,
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        session_id=sess,
        protocol_version=ver,
        tmp_suffix="list",
    )
    assert st == 200, msg
    names = {t["name"] for t in msg.get("result", {}).get("tools", [])}
    assert {"task_health", "create_task", "list_tasks"}.issubset(names)
    for tool in msg["result"]["tools"]:
        if tool["name"] == "create_task":
            props = tool["inputSchema"].get("properties", {})
            assert "chat_id" in props and "assignee" in props
            assert "enum" in props["assignee"]
            break
    else:
        pytest.fail("create_task tool not in list")


@pytest.mark.e2e
def test_mcp_tools_call_task_health(curl_exe, base_url, tmp_path):
    ver, sess = _mcp_bootstrap(curl_exe, base_url, tmp_path)
    st, msg, _ = _mcp_post(
        curl_exe,
        base_url,
        tmp_path,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "task_health", "arguments": {}},
        },
        session_id=sess,
        protocol_version=ver,
        tmp_suffix="call1",
    )
    assert st == 200, msg
    res = msg.get("result", {})
    assert res.get("isError") is False
    sc = res.get("structuredContent", {})
    assert sc.get("status") == "ok"
    assert "sqlite_path" in sc


@pytest.mark.e2e
def test_mcp_create_and_list_tasks(curl_exe, base_url, tmp_path):
    ver, sess = _mcp_bootstrap(curl_exe, base_url, tmp_path)
    st, msg, _ = _mcp_post(
        curl_exe,
        base_url,
        tmp_path,
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "create_task",
                "arguments": {
                    "chat_id": "e2e-chat",
                    "body": "curl e2e task",
                    "assignee": "oksana",
                    "user_id": "e2e-user",
                },
            },
        },
        session_id=sess,
        protocol_version=ver,
        tmp_suffix="create",
    )
    assert st == 200, msg
    sc = msg["result"]["structuredContent"]
    assert sc["chat_id"] == "e2e-chat"
    assert sc["assignee"] == "oksana"
    assert sc["assignee_name"] == "Оксана"
    assert isinstance(sc.get("task_number"), int)

    st2, msg2, _ = _mcp_post(
        curl_exe,
        base_url,
        tmp_path,
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "list_tasks",
                "arguments": {"chat_id": "e2e-chat"},
            },
        },
        session_id=sess,
        protocol_version=ver,
        tmp_suffix="list2",
    )
    assert st2 == 200, msg2
    inner = msg2["result"]["structuredContent"]
    assert inner["count"] >= 1
    assert any(t["body"] == "curl e2e task" for t in inner["tasks"])

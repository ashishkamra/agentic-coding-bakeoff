"""Build OTel trace JSON files in the format inference-perf expects."""

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Any


def make_span(
    trace_id: str,
    span_id: str,
    session_id: str,
    model: str,
    messages: list[dict[str, str]],
    start_time: datetime,
    end_time: datetime,
    prompt_tokens: int,
    completion_tokens: int,
    output_text: str | None = None,
    output_messages: list[dict[str, Any]] | None = None,
    tool_definitions: list[dict[str, Any]] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2000,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "exgentic.session.id": session_id,
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": model,
        "gen_ai.request.max_tokens": max_tokens,
        "gen_ai.request.temperature": temperature,
        "gen_ai.input.messages": json.dumps(messages, ensure_ascii=False),
        "gen_ai.usage.prompt_tokens": prompt_tokens,
        "gen_ai.usage.completion_tokens": completion_tokens,
    }

    if output_messages is not None:
        attrs["gen_ai.output.messages"] = json.dumps(output_messages, ensure_ascii=False)
    elif output_text is not None:
        attrs["gen_ai.output.text"] = output_text
    else:
        attrs["gen_ai.output.text"] = ""

    if tool_definitions is not None:
        attrs["gen_ai.tool.definitions"] = json.dumps(tool_definitions, ensure_ascii=False)

    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": None,
        "name": f"chat {model}",
        "kind": "SPAN_KIND_INTERNAL",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "attributes": attrs,
        "resource_attributes": {"service.name": "coding-agent-benchmark"},
        "status": {"code": 1, "message": ""},
    }


def make_trace(trace_id: str, spans: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "span_count": len(spans),
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "spans": spans,
    }


def write_trace(trace: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)


def iso_time(base: datetime, offset_ms: int) -> datetime:
    return base + timedelta(milliseconds=offset_ms)

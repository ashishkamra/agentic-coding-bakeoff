"""Scenario 1: Long-Context Multi-Turn Agent Loop with Tool Calls.

Generates 6-turn coding-agent sessions where each turn appends to a growing
conversation. All sessions share the same system prompt, tool schemas, and
repo map (enabling prefix cache hits). Sessions differ in bug reports, file
contexts, tool outputs, and patches.

Turns 3 and 5 produce structured tool call outputs (apply_patch).
"""

import json
import os
import random
from datetime import datetime, timezone

from datagen.content import (
    generate_system_prompt,
    generate_tool_schemas,
    generate_repo_map,
    generate_code_files,
    generate_bug_report,
    generate_tool_output,
    generate_patch,
    generate_structured_patch_output,
)
from datagen.tokens import estimate_tokens, pad_to_tokens, generate_prose
from datagen.otel_format import make_span, make_trace, write_trace, iso_time

NUM_SESSIONS = 100

# Shared content (identical across all sessions for prefix caching)
SYSTEM_TOKENS = 2000
TOOL_TOKENS = 8000
REPO_MAP_TOKENS = 40000
FILE_CONTEXT_TOKENS = 15000


def generate(output_dir: str, model: str, seed: int = 42) -> None:
    master_rng = random.Random(seed)

    shared_system = generate_system_prompt(SYSTEM_TOKENS, random.Random(seed))
    tool_text, tool_schemas = generate_tool_schemas(TOOL_TOKENS, random.Random(seed))
    shared_repo_map = generate_repo_map(REPO_MAP_TOKENS, rng=random.Random(seed))

    shared_prefix_tokens = estimate_tokens(shared_system) + estimate_tokens(tool_text) + estimate_tokens(shared_repo_map)

    os.makedirs(output_dir, exist_ok=True)

    for session_idx in range(NUM_SESSIONS):
        session_rng = random.Random(master_rng.randint(0, 2**32))
        trace_id = f"scenario1_session_{session_idx:04d}"
        session_id = f"session_s1_{session_idx:04d}"

        file_context = generate_code_files(FILE_CONTEXT_TOKENS, module_index=session_idx, rng=session_rng)
        bug_report = generate_bug_report(session_idx, rng=session_rng)

        messages: list[dict] = [
            {"role": "system", "content": shared_system},
            {"role": "system", "content": tool_text},
            {"role": "system", "content": shared_repo_map},
            {"role": "user", "content": f"## Active File Context\n\n{file_context}"},
            {"role": "user", "content": bug_report},
        ]

        base_time = datetime(2026, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        spans = []
        time_cursor_ms = session_idx * 120_000  # stagger session starts by 2 min

        # --- Turn 1: Initial analysis ---
        turn1_prompt_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        turn1_output = pad_to_tokens(
            f"I'll analyze the reported issue ACME-{1000 + session_idx}. "
            "Based on the code context and error description, the bug appears to be in the "
            "input validation logic. Let me inspect the relevant test files and logs to confirm "
            "the root cause before proposing a fix.",
            session_rng.randint(800, 1200),
            session_rng,
        )
        turn1_output_tokens = estimate_tokens(turn1_output)

        spans.append(make_span(
            trace_id=trace_id, span_id=f"s1_{session_idx:04d}_turn1",
            session_id=session_id, model=model, messages=list(messages),
            start_time=iso_time(base_time, time_cursor_ms),
            end_time=iso_time(base_time, time_cursor_ms + 3000),
            prompt_tokens=turn1_prompt_tokens, completion_tokens=turn1_output_tokens,
            output_text=turn1_output, tool_definitions=tool_schemas,
            max_tokens=2000,
        ))
        time_cursor_ms += 4000

        # --- Turn 2: Request test/log inspection ---
        messages.append({"role": "assistant", "content": turn1_output})
        turn2_user = pad_to_tokens(
            "Please inspect the test files for this module and check the recent error logs. "
            "I want to see which specific test cases are failing and what the error patterns look like.",
            session_rng.randint(1000, 2000),
            session_rng,
        )
        messages.append({"role": "user", "content": turn2_user})

        turn2_prompt_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        turn2_output = pad_to_tokens(
            "I'll run the test suite for this module and check the logs. "
            "Let me search for the relevant test files and recent error entries.",
            session_rng.randint(500, 800),
            session_rng,
        )
        turn2_output_tokens = estimate_tokens(turn2_output)

        spans.append(make_span(
            trace_id=trace_id, span_id=f"s1_{session_idx:04d}_turn2",
            session_id=session_id, model=model, messages=list(messages),
            start_time=iso_time(base_time, time_cursor_ms),
            end_time=iso_time(base_time, time_cursor_ms + 2000),
            prompt_tokens=turn2_prompt_tokens, completion_tokens=turn2_output_tokens,
            output_text=turn2_output, tool_definitions=tool_schemas,
            max_tokens=2000,
        ))
        time_cursor_ms += 3000

        # --- Turn 3: Receive tool output, generate patch (STRUCTURED OUTPUT) ---
        messages.append({"role": "assistant", "content": turn2_output})
        tool_result = generate_tool_output("test_fail", session_rng.randint(1000, 3000), session_rng)
        messages.append({"role": "user", "content": f"## Test Results\n\n{tool_result}"})

        turn3_prompt_tokens = sum(estimate_tokens(m["content"]) for m in messages)

        patch_text, patch_output_messages, tool_call_id_3 = generate_structured_patch_output(session_idx, session_rng)
        turn3_output_tokens = session_rng.randint(800, 1200)

        spans.append(make_span(
            trace_id=trace_id, span_id=f"s1_{session_idx:04d}_turn3",
            session_id=session_id, model=model, messages=list(messages),
            start_time=iso_time(base_time, time_cursor_ms),
            end_time=iso_time(base_time, time_cursor_ms + 4000),
            prompt_tokens=turn3_prompt_tokens, completion_tokens=turn3_output_tokens,
            output_messages=patch_output_messages, tool_definitions=tool_schemas,
            max_tokens=2000,
        ))
        time_cursor_ms += 5000

        # --- Turn 4: Receive patch result, explain ---
        messages.append({"role": "assistant", "content": patch_text})
        messages.append({"role": "tool", "content": "Patch applied successfully.", "tool_call_id": tool_call_id_3})
        patch_result = generate_tool_output("patch_applied", session_rng.randint(500, 1000), session_rng)
        messages.append({"role": "user", "content": f"## Patch Result\n\n{patch_result}"})

        turn4_prompt_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        turn4_output = pad_to_tokens(
            "The patch has been applied. Let me explain the changes: I fixed the input validation "
            "to properly handle the edge case. Let me now run the test suite to verify the fix.",
            session_rng.randint(500, 1000),
            session_rng,
        )
        turn4_output_tokens = estimate_tokens(turn4_output)

        spans.append(make_span(
            trace_id=trace_id, span_id=f"s1_{session_idx:04d}_turn4",
            session_id=session_id, model=model, messages=list(messages),
            start_time=iso_time(base_time, time_cursor_ms),
            end_time=iso_time(base_time, time_cursor_ms + 2500),
            prompt_tokens=turn4_prompt_tokens, completion_tokens=turn4_output_tokens,
            output_text=turn4_output, tool_definitions=tool_schemas,
            max_tokens=2000,
        ))
        time_cursor_ms += 3500

        # --- Turn 5: Receive failing tests, generate revised patch (STRUCTURED OUTPUT) ---
        messages.append({"role": "assistant", "content": turn4_output})
        failing_tests = generate_tool_output("test_fail", session_rng.randint(2000, 3000), session_rng)
        messages.append({"role": "user", "content": f"## Test Results (after patch)\n\n{failing_tests}"})

        turn5_prompt_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        _, patch_output_messages_2, tool_call_id_5 = generate_structured_patch_output(session_idx + 1000, session_rng)
        turn5_output_tokens = session_rng.randint(1000, 1500)

        revised_patch_text = f"tool_call: apply_patch(files=[{{path: src/..., diff: ...}}]) [revised]"

        spans.append(make_span(
            trace_id=trace_id, span_id=f"s1_{session_idx:04d}_turn5",
            session_id=session_id, model=model, messages=list(messages),
            start_time=iso_time(base_time, time_cursor_ms),
            end_time=iso_time(base_time, time_cursor_ms + 5000),
            prompt_tokens=turn5_prompt_tokens, completion_tokens=turn5_output_tokens,
            output_messages=patch_output_messages_2, tool_definitions=tool_schemas,
            max_tokens=2000,
        ))
        time_cursor_ms += 6000

        # --- Turn 6: Receive passing tests, final explanation ---
        messages.append({"role": "assistant", "content": revised_patch_text})
        messages.append({"role": "tool", "content": "Patch applied successfully.", "tool_call_id": tool_call_id_5})
        passing_tests = generate_tool_output("test_pass", session_rng.randint(500, 1000), session_rng)
        messages.append({"role": "user", "content": f"## Test Results (final)\n\n{passing_tests}"})

        turn6_prompt_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        turn6_output = pad_to_tokens(
            f"All tests pass. The fix for ACME-{1000 + session_idx} is complete. "
            "Summary of changes: I identified the root cause in the input validation logic "
            "and applied a two-part fix. The first patch addressed the primary validation gap, "
            "and the revised patch also handled an edge case revealed by the test suite.",
            session_rng.randint(500, 1000),
            session_rng,
        )
        turn6_output_tokens = estimate_tokens(turn6_output)

        spans.append(make_span(
            trace_id=trace_id, span_id=f"s1_{session_idx:04d}_turn6",
            session_id=session_id, model=model, messages=list(messages),
            start_time=iso_time(base_time, time_cursor_ms),
            end_time=iso_time(base_time, time_cursor_ms + 2000),
            prompt_tokens=turn6_prompt_tokens, completion_tokens=turn6_output_tokens,
            output_text=turn6_output, tool_definitions=tool_schemas,
            max_tokens=2000,
        ))

        trace = make_trace(trace_id, spans)
        write_trace(trace, os.path.join(output_dir, f"session_{session_idx:04d}.json"))

    print(f"Scenario 1: Generated {NUM_SESSIONS} session traces in {output_dir}")
    print(f"  Shared prefix: ~{shared_prefix_tokens} tokens")
    print(f"  Turns per session: 6")

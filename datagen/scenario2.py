"""Scenario 2: Branch / Fan-Out Solution Exploration.

Generates branch sets where a shared coding-agent context forks into 5 parallel
solution attempts. Each trace file contains 6 spans: 1 root coordinator + 5
parallel branches that share the exact same prefix and differ only in the
branch-specific instruction.

inference-perf will detect:
- Causal dependency: branches 1-5 depend on root (root's output in their messages)
- Shared prefix: all branches share the same leading messages (InputSegment type="shared")
- Parallel execution: branches 1-5 have overlapping timestamps
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
)
from datagen.tokens import estimate_tokens, pad_to_tokens, generate_prose
from datagen.otel_format import make_span, make_trace, write_trace, iso_time

NUM_BRANCH_SETS = 50

SYSTEM_TOKENS = 2000
TOOL_TOKENS = 8000
REPO_MAP_TOKENS = 60000
FILE_CONTEXT_TOKENS = 15000
BUG_REPORT_TOKENS = 5000

BRANCH_INSTRUCTIONS = [
    "Produce the smallest possible patch that fixes this bug. Minimize the number of changed lines. "
    "Do not refactor surrounding code. Focus only on the exact root cause and fix it with the "
    "minimum viable change.",

    "Before fixing the bug, refactor the parser boundary and module interface to make the fix "
    "cleaner. Restructure the affected function to separate concerns, then apply the bug fix "
    "within the improved architecture. The refactored code should be more maintainable.",

    "Fix the bug and add comprehensive defensive validation. Add input validation, null checks, "
    "boundary checks, and type assertions. Write at least 5 new test cases covering edge cases "
    "that the current test suite misses. Include tests for empty inputs, boundary values, and "
    "concurrent access patterns.",

    "Fix the bug while also optimizing the hot path for performance. Profile the affected code "
    "and identify any unnecessary allocations, redundant computations, or suboptimal data "
    "structures. The fix should improve both correctness and throughput. Add benchmarks to "
    "demonstrate the performance improvement.",

    "Fix the bug while preserving backward compatibility with the legacy API (v1). The fix must "
    "not change any public API signatures, response formats, or error codes. If the fix requires "
    "API changes, add a compatibility shim that translates between old and new formats. Include "
    "migration notes for consumers of the legacy API.",
]


def generate(output_dir: str, model: str, seed: int = 42, staggered: bool = False) -> None:
    master_rng = random.Random(seed)

    shared_system = generate_system_prompt(SYSTEM_TOKENS, random.Random(seed))
    tool_text, tool_schemas = generate_tool_schemas(TOOL_TOKENS, random.Random(seed))
    shared_repo_map = generate_repo_map(REPO_MAP_TOKENS, rng=random.Random(seed))

    os.makedirs(output_dir, exist_ok=True)

    for set_idx in range(NUM_BRANCH_SETS):
        set_rng = random.Random(master_rng.randint(0, 2**32))
        trace_id = f"scenario2_branchset_{set_idx:04d}"
        session_id = f"session_s2_{set_idx:04d}"

        file_context = generate_code_files(FILE_CONTEXT_TOKENS, module_index=set_idx, rng=set_rng)
        bug_report = generate_bug_report(set_idx, rng=set_rng)
        bug_report = pad_to_tokens(bug_report, BUG_REPORT_TOKENS, set_rng)
        failing_logs = generate_tool_output("test_fail", set_rng.randint(2000, 3000), set_rng)

        # Root messages (shared by all branches)
        root_messages: list[dict] = [
            {"role": "system", "content": shared_system},
            {"role": "system", "content": tool_text},
            {"role": "system", "content": shared_repo_map},
            {"role": "user", "content": f"## Active File Context\n\n{file_context}"},
            {"role": "user", "content": bug_report},
            {"role": "user", "content": f"## Failing Test Logs\n\n{failing_logs}"},
        ]

        shared_prefix_tokens = sum(estimate_tokens(m["content"]) for m in root_messages)

        base_time = datetime(2026, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        time_cursor_ms = set_idx * 60_000

        # --- Root span: coordinator ---
        root_output = pad_to_tokens(
            "I've analyzed the bug report and the failing tests. I'll now explore 5 different "
            "approaches to fixing this issue, each with different trade-offs between minimality, "
            "maintainability, performance, and backward compatibility. Let me generate each "
            "candidate solution.",
            set_rng.randint(300, 600),
            set_rng,
        )
        root_output_tokens = estimate_tokens(root_output)

        spans = [make_span(
            trace_id=trace_id, span_id=f"s2_{set_idx:04d}_root",
            session_id=session_id, model=model, messages=list(root_messages),
            start_time=iso_time(base_time, time_cursor_ms),
            end_time=iso_time(base_time, time_cursor_ms + 2000),
            prompt_tokens=shared_prefix_tokens, completion_tokens=root_output_tokens,
            output_text=root_output, tool_definitions=tool_schemas,
            max_tokens=2000,
        )]
        time_cursor_ms += 2500

        # --- 5 parallel branch spans ---
        for branch_idx, instruction in enumerate(BRANCH_INSTRUCTIONS):
            branch_messages = list(root_messages) + [
                {"role": "assistant", "content": root_output},
                {"role": "user", "content": f"## Approach {chr(65 + branch_idx)}\n\n{instruction}"},
            ]

            branch_prompt_tokens = sum(estimate_tokens(m["content"]) for m in branch_messages)

            solution = pad_to_tokens(
                f"## Solution {chr(65 + branch_idx)}\n\n"
                f"Following the requested approach, here is my implementation:\n\n",
                set_rng.randint(1000, 2000),
                set_rng,
            )
            branch_output_tokens = estimate_tokens(solution)

            if staggered:
                branch_offset_ms = set_rng.randint(0, 5000)
            else:
                branch_offset_ms = branch_idx * 100  # near-simultaneous

            branch_start = time_cursor_ms + branch_offset_ms
            branch_duration = set_rng.randint(3000, 8000)

            spans.append(make_span(
                trace_id=trace_id,
                span_id=f"s2_{set_idx:04d}_branch_{chr(97 + branch_idx)}",
                session_id=session_id, model=model, messages=branch_messages,
                start_time=iso_time(base_time, branch_start),
                end_time=iso_time(base_time, branch_start + branch_duration),
                prompt_tokens=branch_prompt_tokens, completion_tokens=branch_output_tokens,
                output_text=solution, tool_definitions=tool_schemas,
                max_tokens=3000,
            ))

        trace = make_trace(trace_id, spans)
        suffix = "_staggered" if staggered else ""
        write_trace(trace, os.path.join(output_dir, f"branchset_{set_idx:04d}{suffix}.json"))

    variant = "staggered" if staggered else "simultaneous"
    print(f"Scenario 2 ({variant}): Generated {NUM_BRANCH_SETS} branch set traces in {output_dir}")
    print(f"  Shared prefix per branch set: ~{SYSTEM_TOKENS + TOOL_TOKENS + REPO_MAP_TOKENS + FILE_CONTEXT_TOKENS + BUG_REPORT_TOKENS} tokens")
    print(f"  Branches per set: 5")

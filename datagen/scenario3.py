"""Scenario 3: Monorepo Team Swarm with Cache Churn.

Generates many developer sessions against the same large repository with
overlapping but not identical prompt prefixes. Two layout variants:

Variant A (stable): Global prefix always first, then per-session content.
    This maximizes prefix cache effectiveness.

Variant B (noisy): Messages are interleaved (system, task, tools, files, repo).
    This breaks prefix alignment and tests whether prompt order matters.

Sessions use Poisson-distributed arrival times over a 20-minute window with
2-6 turns each. ~90K tokens of shared global prefix.
"""

import json
import math
import os
import random
from datetime import datetime, timezone

from datagen.content import (
    generate_system_prompt,
    generate_tool_schemas,
    generate_repo_map,
    generate_code_files,
    generate_coding_guidelines,
    generate_api_docs,
    generate_bug_report,
    generate_tool_output,
)
from datagen.tokens import estimate_tokens, pad_to_tokens, generate_prose
from datagen.otel_format import make_span, make_trace, write_trace, iso_time

NUM_SESSIONS = 500

# Global shared prefix target: ~90K tokens
SYSTEM_TOKENS = 2000
TOOL_TOKENS = 8000
GUIDELINES_TOKENS = 5000
REPO_MAP_TOKENS = 55000
API_DOCS_TOKENS = 20000

# Per-session
MIN_WORKSPACE_TOKENS = 4000
MAX_WORKSPACE_TOKENS = 20000
MIN_TURNS = 2
MAX_TURNS = 6
WINDOW_SECONDS = 1200  # 20 minutes


def _poisson_arrivals(n: int, window_sec: float, rng: random.Random) -> list[float]:
    """Generate n Poisson-distributed arrival times within window_sec."""
    rate = n / window_sec
    arrivals = []
    t = 0.0
    while len(arrivals) < n:
        t += rng.expovariate(rate)
        if t <= window_sec:
            arrivals.append(t)
        else:
            t = window_sec * rng.random()
            arrivals.append(t)
    arrivals.sort()
    return arrivals[:n]


def _build_stable_messages(
    shared_system: str,
    tool_text: str,
    shared_guidelines: str,
    shared_repo_map: str,
    shared_api_docs: str,
    workspace_files: str,
    task_description: str,
) -> list[dict]:
    """Variant A: Global prefix first, then per-session content."""
    return [
        {"role": "system", "content": shared_system},
        {"role": "system", "content": tool_text},
        {"role": "system", "content": shared_guidelines},
        {"role": "system", "content": shared_repo_map},
        {"role": "system", "content": shared_api_docs},
        {"role": "user", "content": f"## Workspace Files\n\n{workspace_files}"},
        {"role": "user", "content": task_description},
    ]


def _build_noisy_messages(
    shared_system: str,
    tool_text: str,
    shared_guidelines: str,
    shared_repo_map: str,
    shared_api_docs: str,
    workspace_files: str,
    task_description: str,
) -> list[dict]:
    """Variant B: Interleaved messages that break prefix alignment."""
    return [
        {"role": "system", "content": shared_system},
        {"role": "user", "content": task_description},
        {"role": "system", "content": tool_text},
        {"role": "user", "content": f"## Workspace Files\n\n{workspace_files}"},
        {"role": "system", "content": shared_guidelines},
        {"role": "system", "content": shared_repo_map},
        {"role": "system", "content": shared_api_docs},
    ]


def generate(output_dir: str, model: str, seed: int = 42, layout: str = "stable") -> None:
    master_rng = random.Random(seed)

    # Generate shared global content (same seed for reproducibility)
    shared_system = generate_system_prompt(SYSTEM_TOKENS, random.Random(seed))
    tool_text, tool_schemas = generate_tool_schemas(TOOL_TOKENS, random.Random(seed))
    shared_guidelines = generate_coding_guidelines(GUIDELINES_TOKENS, random.Random(seed))
    shared_repo_map = generate_repo_map(REPO_MAP_TOKENS, rng=random.Random(seed))
    shared_api_docs = generate_api_docs(API_DOCS_TOKENS, rng=random.Random(seed))

    global_prefix_tokens = (
        estimate_tokens(shared_system) +
        estimate_tokens(tool_text) +
        estimate_tokens(shared_guidelines) +
        estimate_tokens(shared_repo_map) +
        estimate_tokens(shared_api_docs)
    )

    build_messages = _build_stable_messages if layout == "stable" else _build_noisy_messages

    arrivals = _poisson_arrivals(NUM_SESSIONS, WINDOW_SECONDS, random.Random(seed + 1))

    os.makedirs(output_dir, exist_ok=True)

    for session_idx in range(NUM_SESSIONS):
        session_rng = random.Random(master_rng.randint(0, 2**32))
        trace_id = f"scenario3_{layout}_session_{session_idx:04d}"
        session_id = f"session_s3_{layout}_{session_idx:04d}"

        # Per-session workspace files (drawn from module pool with overlap)
        workspace_tokens = session_rng.randint(MIN_WORKSPACE_TOKENS, MAX_WORKSPACE_TOKENS)
        module_idx = session_idx % 10  # 10 module pools → 20-40% overlap
        workspace_files = generate_code_files(workspace_tokens, module_index=module_idx, rng=session_rng)

        num_turns = session_rng.randint(MIN_TURNS, MAX_TURNS)
        task_description = generate_bug_report(session_idx, rng=session_rng)

        messages = build_messages(
            shared_system, tool_text, shared_guidelines,
            shared_repo_map, shared_api_docs,
            workspace_files, task_description,
        )

        base_time = datetime(2026, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        session_start_ms = int(arrivals[session_idx] * 1000)

        spans = []
        time_cursor_ms = session_start_ms

        for turn_idx in range(num_turns):
            prompt_tokens = sum(estimate_tokens(m["content"]) for m in messages)

            output_tokens = session_rng.randint(500, 1500)
            output_text = pad_to_tokens(
                f"Turn {turn_idx + 1} analysis for session {session_idx}. ",
                output_tokens,
                session_rng,
            )
            actual_output_tokens = estimate_tokens(output_text)

            duration_ms = session_rng.randint(2000, 6000)

            spans.append(make_span(
                trace_id=trace_id,
                span_id=f"s3_{layout}_{session_idx:04d}_turn{turn_idx + 1}",
                session_id=session_id, model=model, messages=list(messages),
                start_time=iso_time(base_time, time_cursor_ms),
                end_time=iso_time(base_time, time_cursor_ms + duration_ms),
                prompt_tokens=prompt_tokens, completion_tokens=actual_output_tokens,
                output_text=output_text, tool_definitions=tool_schemas,
                max_tokens=2000,
            ))

            # Prepare next turn
            messages.append({"role": "assistant", "content": output_text})

            if turn_idx < num_turns - 1:
                tool_kind = session_rng.choice(["test_fail", "test_pass", "grep", "diagnostics"])
                tool_tokens = session_rng.randint(1000, 5000)
                tool_result = generate_tool_output(tool_kind, tool_tokens, session_rng)
                messages.append({"role": "user", "content": f"## Tool Output\n\n{tool_result}"})

            inter_turn_delay = session_rng.randint(2000, 8000)
            time_cursor_ms += duration_ms + inter_turn_delay

        trace = make_trace(trace_id, spans)
        write_trace(trace, os.path.join(output_dir, f"session_{session_idx:04d}.json"))

    print(f"Scenario 3 ({layout} layout): Generated {NUM_SESSIONS} session traces in {output_dir}")
    print(f"  Global shared prefix: ~{global_prefix_tokens} tokens")
    print(f"  Turns per session: {MIN_TURNS}-{MAX_TURNS}")
    print(f"  Arrival window: {WINDOW_SECONDS}s")

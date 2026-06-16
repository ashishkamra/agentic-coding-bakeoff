"""Validate generated traces by running inference-perf's graph builder directly.

Imports only the graph building and trace parsing code (no server/client deps).
"""

import json
import sys
import os
import importlib

_INFPERF_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "inference-perf")

# Avoid triggering inference_perf.__init__ (which imports heavy deps like av).
# Instead, load the specific modules we need directly.
def _load_module(dotted_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(dotted_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod

# Pre-register leaf deps so their imports resolve
_load_module(
    "inference_perf.datagen.replay_graph_types",
    os.path.join(_INFPERF_ROOT, "inference_perf/datagen/replay_graph_types.py"),
)
_load_module(
    "inference_perf.datagen.otel_trace_utils",
    os.path.join(_INFPERF_ROOT, "inference_perf/datagen/otel_trace_utils.py"),
)
_load_module(
    "inference_perf.datagen.export_replay_graph_to_dot",
    os.path.join(_INFPERF_ROOT, "inference_perf/datagen/export_replay_graph_to_dot.py"),
)

_graph_mod = _load_module(
    "inference_perf.datagen.otel_trace_to_replay_graph",
    os.path.join(_INFPERF_ROOT, "inference_perf/datagen/otel_trace_to_replay_graph.py"),
)

build_raw_calls = _graph_mod.build_raw_calls
build_graph = _graph_mod.build_graph
print_graph = _graph_mod.print_graph
graph_to_dict = _graph_mod.graph_to_dict
summarize_graph = _graph_mod.summarize_graph


def validate_trace(trace_path: str, verbose: bool = False) -> dict:
    with open(trace_path) as f:
        data = json.load(f)

    spans = data.get("spans", [])
    if not spans:
        print(f"  ERROR: No spans in {trace_path}")
        return {"ok": False, "error": "no spans"}

    calls = build_raw_calls(spans)
    if not calls:
        print(f"  ERROR: No LLM spans extracted from {trace_path}")
        return {"ok": False, "error": "no LLM spans"}

    graph = build_graph(calls, source_file=trace_path)

    event_count = len(graph.events)
    root_count = len(graph.root_event_ids)

    # Check segments
    total_shared = 0
    total_output = 0
    total_unique = 0
    for event in graph.events.values():
        for seg in event.call.input_segments:
            if seg.type == "shared":
                total_shared += seg.token_count
            elif seg.type == "output":
                total_output += seg.token_count
            elif seg.type == "unique":
                total_unique += seg.token_count

    result = {
        "ok": True,
        "events": event_count,
        "roots": root_count,
        "shared_tokens": total_shared,
        "output_tokens": total_output,
        "unique_tokens": total_unique,
    }

    if verbose:
        print_graph(graph)

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate generated OTel traces")
    parser.add_argument("--trace", help="Specific trace file to validate")
    parser.add_argument("--scenario", choices=["1", "2", "3"], help="Validate sample from scenario")
    parser.add_argument("--verbose", action="store_true", help="Print full graph")
    parser.add_argument("--traces-dir", default="./traces", help="Root traces directory")
    args = parser.parse_args()

    if args.trace:
        print(f"Validating: {args.trace}")
        result = validate_trace(args.trace, verbose=args.verbose)
        print(f"  Result: {json.dumps(result, indent=2)}")
        return

    scenarios = [args.scenario] if args.scenario else ["1", "2", "3"]

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO {scenario}")
        print(f"{'='*60}")

        if scenario == "1":
            trace_dir = os.path.join(args.traces_dir, "scenario1")
            samples = ["session_0000.json", "session_0050.json", "session_0099.json"]
            expected_events = 6
            expected_roots = 1

        elif scenario == "2":
            trace_dir = os.path.join(args.traces_dir, "scenario2_sync")
            samples = ["branchset_0000.json", "branchset_0025.json", "branchset_0049.json"]
            expected_events = 6
            expected_roots = 1

        elif scenario == "3":
            trace_dir = os.path.join(args.traces_dir, "scenario3_stable")
            samples = ["session_0000.json", "session_0250.json", "session_0499.json"]
            expected_events = None  # varies 2-6
            expected_roots = 1

        for sample in samples:
            path = os.path.join(trace_dir, sample)
            if not os.path.exists(path):
                print(f"  SKIP: {path} not found")
                continue

            print(f"\n  {sample}:")
            result = validate_trace(path, verbose=args.verbose)

            if not result["ok"]:
                print(f"    FAIL: {result.get('error')}")
                continue

            checks = []
            if expected_events is not None and result["events"] != expected_events:
                checks.append(f"events={result['events']} (expected {expected_events})")
            if result["roots"] != expected_roots:
                checks.append(f"roots={result['roots']} (expected {expected_roots})")
            if result["shared_tokens"] == 0 and scenario != "3":
                checks.append("WARNING: no shared segments detected")

            status = "PASS" if not checks else "WARN"
            print(f"    {status}: {result['events']} events, {result['roots']} root(s)")
            print(f"    Segments: shared={result['shared_tokens']}t, output={result['output_tokens']}t, unique={result['unique_tokens']}t")
            if checks:
                for c in checks:
                    print(f"    ! {c}")


if __name__ == "__main__":
    main()

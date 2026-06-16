"""CLI entry point: generate all trace files and inference-perf configs.

Usage:
    python -m datagen.generate_all --model meta-llama/Llama-3.3-70B-Instruct
    python -m datagen.generate_all --model Qwen/Qwen3-32B --seed 123 --output-dir ./out
"""

import argparse
import os
import yaml

from datagen import scenario1, scenario2, scenario3


def _write_config(path: str, config: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _base_config(model: str, server_type: str = "vllm") -> dict:
    return {
        "api": {"type": "chat"},
        "server": {
            "type": server_type,
            "model_name": model,
            "base_url": "http://localhost:8000",
            "ignore_eos": True,
        },
        "metrics": {
            "type": "prometheus",
            "prometheus": {
                "scrape_url": "http://localhost:9090/api/v1/query_range",
            },
        },
        "report": {
            "request_lifecycle": {
                "summary": True,
                "per_stage": True,
                "per_request": True,
            },
        },
    }


def _otel_data_config(trace_dir: str, model: str) -> dict:
    return {
        "data": {
            "type": "otel_trace_replay",
            "otel_trace_replay": {
                "trace_directory": trace_dir,
                "use_static_model": True,
                "static_model_name": model,
            },
        },
    }


def _session_load_config(concurrent_sessions: int, num_sessions: int | None = None) -> dict:
    stage: dict = {"concurrent_sessions": concurrent_sessions}
    if num_sessions is not None:
        stage["num_sessions"] = num_sessions
    return {
        "load": {
            "type": "trace_session_replay",
            "stages": [stage],
            "num_workers": 8,
            "worker_max_concurrency": 1000,
        },
    }


def generate_scenario1_configs(config_dir: str, trace_dir: str, model: str) -> None:
    out = os.path.join(config_dir, "scenario1")
    for concurrency in [1, 10, 25, 50, 100]:
        config = {
            **_base_config(model),
            **_otel_data_config(trace_dir, model),
            **_session_load_config(concurrency, num_sessions=100),
        }
        _write_config(os.path.join(out, f"c{concurrency}.yml"), config)
    print(f"  Scenario 1 configs: {out}/c{{1,10,25,50,100}}.yml")


def generate_scenario2_configs(config_dir: str, trace_dir_sync: str, trace_dir_stagger: str, model: str) -> None:
    out = os.path.join(config_dir, "scenario2")
    for branch_sets in [10, 25, 50]:
        num_sessions = branch_sets  # each session = 1 branch set (6 spans)
        for variant, trace_dir in [("sync", trace_dir_sync), ("stagger", trace_dir_stagger)]:
            config = {
                **_base_config(model),
                **_otel_data_config(trace_dir, model),
                **_session_load_config(branch_sets, num_sessions=num_sessions),
            }
            _write_config(os.path.join(out, f"c{branch_sets}_{variant}.yml"), config)
    print(f"  Scenario 2 configs: {out}/c{{10,25,50}}_{{sync,stagger}}.yml")


def generate_scenario3_configs(config_dir: str, trace_dir_stable: str, trace_dir_noisy: str, model: str) -> None:
    out = os.path.join(config_dir, "scenario3")
    for num_sessions in [100, 250, 500]:
        concurrent = min(num_sessions, 50)
        for variant, trace_dir in [("stable", trace_dir_stable), ("noisy", trace_dir_noisy)]:
            config = {
                **_base_config(model),
                **_otel_data_config(trace_dir, model),
                **_session_load_config(concurrent, num_sessions=num_sessions),
            }
            _write_config(os.path.join(out, f"c{num_sessions}_{variant}.yml"), config)
    print(f"  Scenario 3 configs: {out}/c{{100,250,500}}_{{stable,noisy}}.yml")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all trace files and inference-perf configs for the vLLM vs SGLang bakeoff.",
    )
    parser.add_argument("--model", required=True, help="Model name (e.g. meta-llama/Llama-3.3-70B-Instruct)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", default="./traces", help="Root directory for trace files")
    parser.add_argument("--config-dir", default="./configs", help="Root directory for YAML configs")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")
    print()

    # --- Generate traces ---
    print("=== Generating trace files ===")

    s1_dir = os.path.join(args.output_dir, "scenario1")
    scenario1.generate(s1_dir, args.model, args.seed)
    print()

    s2_sync_dir = os.path.join(args.output_dir, "scenario2_sync")
    scenario2.generate(s2_sync_dir, args.model, args.seed, staggered=False)

    s2_stagger_dir = os.path.join(args.output_dir, "scenario2_stagger")
    scenario2.generate(s2_stagger_dir, args.model, args.seed, staggered=True)
    print()

    s3_stable_dir = os.path.join(args.output_dir, "scenario3_stable")
    scenario3.generate(s3_stable_dir, args.model, args.seed, layout="stable")

    s3_noisy_dir = os.path.join(args.output_dir, "scenario3_noisy")
    scenario3.generate(s3_noisy_dir, args.model, args.seed, layout="noisy")
    print()

    # --- Generate configs ---
    print("=== Generating inference-perf configs ===")
    generate_scenario1_configs(args.config_dir, s1_dir, args.model)
    generate_scenario2_configs(args.config_dir, s2_sync_dir, s2_stagger_dir, args.model)
    generate_scenario3_configs(args.config_dir, s3_stable_dir, s3_noisy_dir, args.model)

    print()
    print("Done. To validate a trace:")
    print(f"  cd /home/akamra/workspace/inference-perf")
    print(f"  python -m inference_perf.datagen.otel_trace_to_replay_graph \\")
    print(f"    --input ../{args.output_dir}/scenario1/session_0000.json \\")
    print(f"    --output /tmp/graph.json --summary")


if __name__ == "__main__":
    main()

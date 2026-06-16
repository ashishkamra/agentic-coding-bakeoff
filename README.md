# vLLM vs SGLang: Coding-Agent Workload Bakeoff

Benchmark dataset generator for comparing [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) on realistic coding-agent workloads. Generates [OpenTelemetry trace](https://opentelemetry.io/) files and configuration for [inference-perf](https://github.com/kubernetes-sigs/inference-perf)'s agentic trace replay feature.

## Why This Benchmark

Standard LLM benchmarks use single-turn, independent prompts. Coding agents are different — they send **long, growing prompts** with large stable prefixes, multi-turn tool-call loops, parallel branching, and overlapping shared context across users. These patterns stress KV cache management in ways that generic benchmarks miss.

This toolkit generates three scenarios that isolate the cache behaviors that matter most for coding-agent deployments:

| Scenario | Tests | Key Cache Behavior |
|:---------|:------|:-------------------|
| **1. Multi-Turn Agent Loop** | 6-turn debug-fix-verify loop | Prefix reuse across growing turns |
| **2. Branch / Fan-Out** | 5 parallel solution attempts from shared context | Tree-shaped prefix sharing |
| **3. Monorepo Team Swarm** | 100-500 concurrent developer sessions | Cache churn under shared-prefix pressure |

## Quick Start

### Prerequisites

- Python 3.10+
- [inference-perf](https://github.com/kubernetes-sigs/inference-perf) (for running benchmarks)
- A running vLLM or SGLang server with your target model
- PyYAML (`pip install pyyaml`)

### Generate the Dataset

```bash
git clone https://github.com/ashishkamra/agentic-coding-bakeoff.git
cd agentic-coding-bakeoff

# Generate all traces and configs for your model
python -m datagen.generate_all --model meta-llama/Llama-3.3-70B-Instruct

# Or with custom options
python -m datagen.generate_all \
    --model Qwen/Qwen3-32B \
    --seed 123 \
    --output-dir ./traces \
    --config-dir ./configs
```

This generates:
- **1,200 trace files** across 5 directories (~2.2 GB)
- **17 YAML config files** for inference-perf

### Validate the Traces

```bash
# Validate samples from all three scenarios
python -m datagen.validate --traces-dir ./traces

# Verbose output for a single trace (shows full dependency graph)
python -m datagen.validate --trace ./traces/scenario1/session_0000.json --verbose
```

### Run a Benchmark

```bash
# Start your model server (vLLM example)
vllm serve meta-llama/Llama-3.3-70B-Instruct --port 8000

# Run scenario 1 at concurrency 25
cd /path/to/inference-perf
python -m inference_perf --config /path/to/agentic-coding-bakeoff/configs/scenario1/c25.yml

# Compare with SGLang: edit the config to set server.type: sglang, or override at CLI
```

## Scenarios

### Scenario 1: Multi-Turn Agent Loop with Tool Calls

Simulates an interactive coding assistant that reads code, inspects tests, calls tools, generates patches, and iterates.

**Workload shape:**

```
Turn 1: [System 2K + Tools 8K + Repo 40K + Files 15K + Task 1K] → Analysis (800-1200 tokens)
Turn 2: [+ assistant output + "inspect tests" request]           → Inspection plan (500-800 tokens)
Turn 3: [+ assistant output + test logs 1-3K]                    → apply_patch tool call (structured output)
Turn 4: [+ assistant output + patch result]                      → Explanation (500-1000 tokens)
Turn 5: [+ assistant output + failing tests 2-3K]                → Revised apply_patch (structured output)
Turn 6: [+ assistant output + passing tests]                     → Final summary (500-1000 tokens)
```

**What it measures:**
- TTFT across turns 1-6 (does prefill cost drop as prefix is reused?)
- Cache hit ratio for the stable 65K-token prefix
- Structured output overhead (TPOT on tool-call turns vs plain text)
- KV cache retention under concurrency pressure

**Concurrency levels:** 1, 10, 25, 50, 100 concurrent sessions

**Key design:** All 100 sessions share the **exact same** system prompt, tool schemas, and repository map. Only the bug reports, file contexts, and tool outputs differ. This creates a large shared prefix that both vLLM APC and SGLang radix-tree should cache — the benchmark measures *how well* they do it.

**Configs:** `configs/scenario1/c{1,10,25,50,100}.yml`

---

### Scenario 2: Branch / Fan-Out Solution Exploration

Simulates a coding agent that forks one shared context into 5 parallel candidate fixes.

**Workload shape:**

```
Root:     [System + Tools + Repo + Bug Report + Logs = 55-95K tokens] → Coordinator response
Branch A: [Root prefix + Root output + "smallest patch"]              → Solution A (1-2K tokens)
Branch B: [Root prefix + Root output + "refactor first"]              → Solution B (1-2K tokens)
Branch C: [Root prefix + Root output + "add validation"]              → Solution C (1-2K tokens)
Branch D: [Root prefix + Root output + "optimize hot path"]           → Solution D (1-2K tokens)
Branch E: [Root prefix + Root output + "preserve compat"]             → Solution E (1-2K tokens)
```

**What it measures:**
- TTFT for first branch vs subsequent siblings (does the engine reuse the root prefix?)
- Incremental KV memory per branch (is the shared prefix stored once or five times?)
- Cache eviction behavior when branch sets overlap
- Simultaneous vs staggered arrival performance

**Concurrency levels:** 10, 25, 50 branch sets (= 60, 150, 300 total requests)

**Key design:** All 5 branches share the **exact same** prefix messages and differ only in the final user instruction. inference-perf's graph builder decomposes each branch input into `SHARED(~93K tokens) → OUTPUT(~500 tokens) → UNIQUE(~55 tokens)`. This is the clearest possible test of branch-aware prefix sharing.

**Configs:** `configs/scenario2/c{10,25,50}_{sync,stagger}.yml`

---

### Scenario 3: Monorepo Team Swarm with Cache Churn

Simulates 100-500 developers working against the same large monorepo simultaneously.

**Workload shape:**

```
Shared global prefix (~90K tokens):
  System prompt (2K) + Tool schemas (8K) + Coding guidelines (5K) +
  Repository map (55K) + API docs (20K)

Per-session delta (varies):
  Workspace files (4-20K) + Task description (0.5-2K) +
  2-6 turns with tool outputs (1-5K each)
```

**What it measures:**
- Cache hit ratio for the shared 90K-token global prefix under churn
- Eviction rate when KV cache approaches saturation
- TTFT p50/p95/p99 over a 20-minute continuous run
- **Layout sensitivity**: Does prompt message ordering affect cache effectiveness?

**Layout variants:**

| Variant | Message Order | Cache Behavior |
|:--------|:-------------|:---------------|
| **Stable** | Global prefix first → workspace → task → turns | Maximizes prefix alignment |
| **Noisy** | System → task → tools → files → repo → turns | Breaks prefix alignment |

If Variant A performs much better than Variant B on *both* engines, the coding harness needs prompt canonicalization before drawing conclusions about engine differences.

**Concurrency levels:** 100, 250, 500 sessions (Poisson arrival over 20 minutes)

**Configs:** `configs/scenario3/c{100,250,500}_{stable,noisy}.yml`

## Primary Metrics

For each scenario, collect:

| Metric | What It Reveals |
|:-------|:---------------|
| **TTFT p50/p95/p99** (by turn) | Prefill latency; shows cache miss cliffs |
| **Cache hit ratio** | Fraction of prompt tokens served from KV cache |
| **KV memory usage over time** | Whether shared prefixes are deduplicated |
| **Eviction count** | How aggressively the engine discards reusable context |
| **TPOT / ITL** | Decode speed; structured output overhead on tool-call turns |
| **Throughput per GPU-hour** | Completed coding tasks at fixed SLO |
| **p95 TTFT under load** | Tail latency stability as concurrency increases |

## Repository Structure

```
agentic-coding-bakeoff/
├── README.md
├── .gitignore
├── datagen/                    # Dataset generation toolkit
│   ├── __init__.py
│   ├── generate_all.py         # CLI entry point
│   ├── content.py              # Content templates (prompts, code, bugs, patches)
│   ├── tokens.py               # Token counting & text padding
│   ├── otel_format.py          # OTel trace JSON builders
│   ├── scenario1.py            # Multi-turn agent loop generator
│   ├── scenario2.py            # Branch/fan-out generator
│   ├── scenario3.py            # Monorepo swarm generator
│   └── validate.py             # Trace validation using inference-perf's graph builder
├── configs/                    # inference-perf YAML configs
│   ├── scenario1/              # c{1,10,25,50,100}.yml
│   ├── scenario2/              # c{10,25,50}_{sync,stagger}.yml
│   └── scenario3/              # c{100,250,500}_{stable,noisy}.yml
└── traces/                     # Generated traces (gitignored, ~2.2 GB)
    ├── scenario1/              # 100 session traces (6 turns each)
    ├── scenario2_sync/         # 50 branch sets (simultaneous)
    ├── scenario2_stagger/      # 50 branch sets (1-5s staggered)
    ├── scenario3_stable/       # 500 sessions (prefix-first layout)
    └── scenario3_noisy/        # 500 sessions (interleaved layout)
```

## How It Works

### Trace Format

Each trace is an [OpenTelemetry](https://opentelemetry.io/) JSON file with `gen_ai.*` semantic conventions:

```json
{
  "trace_id": "scenario1_session_0042",
  "span_count": 6,
  "spans": [
    {
      "span_id": "s1_0042_turn1",
      "name": "chat meta-llama/Llama-3.3-70B-Instruct",
      "start_time": "2026-06-15T10:00:00+00:00",
      "end_time": "2026-06-15T10:00:03+00:00",
      "attributes": {
        "exgentic.session.id": "session_s1_0042",
        "gen_ai.request.model": "meta-llama/Llama-3.3-70B-Instruct",
        "gen_ai.input.messages": "[{\"role\": \"system\", \"content\": \"...\"}, ...]",
        "gen_ai.output.text": "I'll analyze the reported issue...",
        "gen_ai.usage.prompt_tokens": 65000,
        "gen_ai.usage.completion_tokens": 1000,
        "gen_ai.tool.definitions": "[...]"
      }
    }
  ]
}
```

### Dependency Detection

inference-perf automatically reconstructs the dependency graph by matching output text between spans:

- **Causal edges**: When span B's input messages contain span A's output text
- **Temporal edges**: Fallback ordering by timestamp when no causal link exists
- **Parallel detection**: Spans with overlapping timestamps run concurrently

The generator carefully constructs messages so that:
- **Scenario 1** produces a 6-event **linear chain** (each turn includes the previous output)
- **Scenario 2** produces a **1-root + 5-branch tree** (all branches include root's output)
- **Scenario 3** produces **independent sessions** with shared prefix detection

### Prefix Sharing

inference-perf decomposes each span's input into segments:

| Segment Type | Meaning | Cache Implication |
|:-------------|:--------|:-----------------|
| `SHARED` | Leading messages identical to a predecessor | KV cache hit (no recomputation) |
| `OUTPUT` | Assistant message from a predecessor's output | Injected at replay time |
| `UNIQUE` | Messages unique to this span | Must be computed fresh |

Example from Scenario 2 (branch A):
```
SHARED(6 messages / 92,890 tokens)  ← exact prefix from root
OUTPUT(1 message / 539 tokens)      ← root's output
UNIQUE(1 message / 55 tokens)       ← "produce the smallest patch"
```

## Configuration

### Switching Between vLLM and SGLang

Each config defaults to `server.type: vllm`. To benchmark SGLang, either:

1. Edit the YAML:
   ```yaml
   server:
     type: sglang
   ```

2. Or regenerate configs (modify `generate_all.py`'s `_base_config` function)

### Adjusting Concurrency

Edit the `load.stages` section:

```yaml
load:
  type: trace_session_replay
  stages:
    - concurrent_sessions: 25    # Max sessions running simultaneously
      num_sessions: 100          # Total sessions to complete
```

### Custom Model

The `--model` flag sets both the model name in trace spans and the `server.model_name` in configs. It must match the model served by your vLLM/SGLang instance.

### Prometheus Metrics

Configs expect Prometheus at `http://localhost:9090`. Adjust `metrics.prometheus.scrape_url` if your setup differs. Set `metrics.type: default` to skip Prometheus and use only request-lifecycle metrics.

## Token Budget

| Scenario | Traces | Shared Prefix | Per-Session Unique | Turns | Est. Total Input |
|:---------|:-------|:-------------|:-------------------|:------|:----------------|
| 1 | 100 | ~60K | ~10K | 6 | ~7.2M tokens |
| 2 | 50 sets × 6 | ~93K | ~1K per branch | 1 | ~22.8M tokens |
| 3 | 500 × 2 layouts | ~90K | ~15K | 2-6 | ~55M tokens |

## Reproducibility

All content generation is seeded (`--seed 42` by default). The same seed produces identical traces, enabling A/B comparisons between engine versions, configurations, or hardware.

```bash
# These produce identical output
python -m datagen.generate_all --model my-model --seed 42
python -m datagen.generate_all --model my-model --seed 42

# Different seed → different bug reports, file contexts, patches (same structure)
python -m datagen.generate_all --model my-model --seed 123
```

## License

This project generates synthetic benchmark data. The generated traces contain no real code, user data, or proprietary information.

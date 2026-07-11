# toks-bench Methodology

This document describes how `toks-bench` measures token throughput and latency
for OpenAI-compatible LLM inference servers.

## What we measure

`toks-bench` issues chat-completion requests to a configured provider and records
per-token timing from the client side. All measurements are taken from the
benchmark host, so they reflect the end-to-end latency visible to a caller:
network serialization, queueing, prefill, and token generation.

### Generation mode (`generate`)

For each configured prompt, the tool sends `runs` streaming requests to the
provider and records:

- **TTFT** (time to first token): elapsed milliseconds from request start until
the first content chunk arrives.
- **TPOT** (time per output token): `(total_ms - ttft_ms) / (output_tokens - 1)`
for runs that produced more than one token.
- **Throughput**: `output_tokens / (total_ms / 1000)` tokens per second.
- **Output length**: completion tokens reported by the server, or the count of
content deltas if the server omits usage data.

### Tool mode (`tool`)

For prompts marked as tool prompts in `config.yaml`, the tool sends
non-streaming requests with the configured tool schema and records the total
round-trip latency plus the number of tool calls returned.

## Request parameters

Default parameters live in `config.yaml` under `defaults`:

```yaml
defaults:
  runs: 10
  max_tokens: 512
  temperature: 0.7
  top_p: 0.9
```

You can override them on the command line:

```bash
toks-bench --provider vllm-qwen-fp8 --prompt short --runs 5 --max-tokens 256
```

## Prompts

Prompts are defined in `config.yaml`. The stock prompts exercise different
context lengths and workloads:

| Prompt | Type | Purpose |
|--------|------|---------|
| `short` | generate | One-sentence generation, measures best-case TTFT |
| `medium` | generate | Short article summary, light context |
| `long` | generate | Multi-paragraph summary, long context prefill |
| `code` | generate | Python function generation |
| `tool` | tool | Function-calling round-trip latency |

## Providers

A provider is any OpenAI-compatible chat-completions endpoint. The stock
configuration includes local `llama-server`, Ollama, vLLM, and TensorRT-LLM
endpoints. Add or remove providers by editing `config.yaml` without changing
code.

## Timeouts and failure handling

Two independent timeouts protect the runner from hanging servers:

1. **Client HTTP timeout** (`timeout=30.0` in `toks_bench/providers.py`): the
   OpenAI/httpx client aborts an individual read operation if no data arrives
   for 30 seconds.
2. **Per-run wall-clock timeout** (`_RUN_TIMEOUT_SECONDS = 180.0` in
   `toks_bench/bench.py`): a `SIGALRM` backstop aborts the entire run if it has
   not finished within 180 seconds. This catches servers that keep a streaming
   connection open after emitting the final chunk.

If a single run fails or times out, the runner records a failed run with zero
tokens, prints a warning, and continues with the remaining runs. This keeps a
sweep against many providers from stopping because one endpoint is slow or
down.

## Running a full sweep

Start all configured providers, then run:

```bash
cd toks-bench
source .venv/bin/activate
for prompt in short medium long tool; do
  toks-bench --prompt "$prompt" --all --format json \
    --output "results/full/all-${prompt}.json"
done
toks-bench-aggregate results/full
```

For a long-running sweep, run it in a tmux session so it survives disconnection:

```bash
tmux new-session -d -s bench
# or attach to the existing bench-vllm-nematron session
tmux send-keys -t bench 'source .venv/bin/activate && ...' Enter
```

## Output format

JSON output contains an aggregate object plus per-run details:

```json
{
  "provider-name": {
    "aggregate": {
      "runs": 10,
      "tok_per_sec_mean": 123.4,
      "tok_per_sec_median": 125.0,
      "tok_per_sec_p95": 95.0,
      "ttft_ms_mean": 45.0,
      "tpot_ms_mean": 8.0,
      "finish_reasons": {"length": 8, "stop": 2}
    },
    "runs": [...]
  }
}
```

Use `toks-bench-aggregate` to turn a directory of JSON files into a CSV table
and Markdown report.

## Interpreting results

- **TTFT** is dominated by prompt prefill and queueing; it tends to grow with
  prompt length and batch size.
- **TPOT** reflects per-token generation latency; improvements here come from
  faster kernels, lower batching, or quantization.
- **tok/s** is the headline throughput metric. Higher is better, but compare at
  the same `max_tokens` and prompt because output length affects the average.
- **finish_reasons** tells you how the server ended each run. A prevalence of
  `length` means generations were truncated by `max_tokens`; `stop` means the
  model emitted an end-of-sequence token.

## Reproducibility checklist

- Use the same `config.yaml` across runs.
- Keep the server warm; cold starts can skew the first run.
- Run enough iterations to average out jitter (`runs >= 10` is recommended).
- Ensure no other heavy workloads are sharing the GPU during benchmarking.
- Verify endpoints are responding with `curl http://localhost:<port>/v1/models`
  before starting a sweep.

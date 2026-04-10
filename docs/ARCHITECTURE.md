# CrashSafe Architecture

## Design Goals

1. **Correctness**: Recovery must produce the same final output as uninterrupted generation.
2. **Measurability**: Every component emits structured metrics for experimental analysis.
3. **Modularity**: Backends, recovery strategies, and storage are independently swappable.
4. **Reproducibility**: Deterministic mock backend ensures experiments are reproducible.
5. **CPU-compatibility**: All core functionality runs without GPU access.

---

## Component Overview

### 1. Backend Layer (`src/backends/`)

The backend layer abstracts over different LLM inference engines behind a unified interface.

**`BaseBackend`** defines two core methods:
- `generate(request) → InferenceResponse`: blocking full-sequence generation
- `stream_generate(request) → AsyncIterator[Token]`: async streaming

**`InferenceRequest`** carries the input prompt and, crucially, `resume_from_tokens`: the list of token IDs already committed before a fault. When set, the backend skips recomputing those tokens and resumes from that position.

**`MockBackend`** generates tokens deterministically via `MD5(prompt:position) % vocab_size`, producing reproducible experiments without any real model. It simulates configurable generation speed (tokens/second) and base latency (TTFT).

**`TransformersBackend`** wraps HuggingFace `AutoModelForCausalLM` for real inference on CPU or GPU. Streaming is implemented via `TextIteratorStreamer` running in a background thread.

**`VLLMBackend`** wraps the `AsyncLLMEngine` for high-throughput PagedAttention-based inference (GPU only).

---

### 2. Recovery Layer (`src/recovery/`)

Recovery strategies implement `BaseRecoveryStrategy.execute(context, generate_fn) → RecoveryResult`.

The `generate_fn` callable represents a single generation attempt. The strategy decides how many times to call it and what request to pass on each attempt.

#### FAIL_STOP

```
attempt generate_fn(original_request)
  success → return RecoveryResult(success=True, retries=0)
  failure → return RecoveryResult(success=False, retries=0)
```

No retry. Zero overhead. Acts as the lower bound.

#### RETRY_FROM_SCRATCH

```
for attempt in range(max_retries + 1):
  try generate_fn(original_request)
    success → return
  except:
    sleep(backoff_base * 2^attempt)
    recomputed += max_new_tokens
```

Each failed attempt wastes `max_new_tokens` of compute. Recovery time grows linearly with retries.

#### TOKEN_COMMIT_RESUME

```
initialize checkpoint(request_id)
for attempt in range(max_retries + 1):
  checkpoint = load_checkpoint(request_id)
  resume_request = build_resume_request(original, checkpoint.token_ids)
  try generate_fn(resume_request)
    tokens = response.tokens
    checkpoint_tokens(tokens[len(committed):])  # only new tokens
    delete_checkpoint(request_id)
    return RecoveryResult(success=True, recomputed=max_tokens-len(committed))
  except:
    committed = len(checkpoint.tokens)  # how far we got
    sleep(backoff)
```

Recovery time is proportional to `max_new_tokens - committed`, not `max_new_tokens`.

---

### 3. Storage Layer (`src/storage/`)

**`DiskTokenStore`**: JSONL-based file storage. Each request gets a file:

```
checkpoints/req-abc123.jsonl
{"type": "header", "request_id": "req-abc123", "prompt": "...", "max_new_tokens": 128}
{"type": "token", "position": 0, "token_id": 42, "token_text": "the"}
{"type": "token", "position": 1, "token_id": 17, "token_text": "model"}
...
```

Lines are appended atomically per batch. On resume, the file is parsed; incomplete lines (from mid-write crashes) are skipped. Tokens are sorted by position before returning the checkpoint.

**`MemoryTokenStore`**: In-memory dictionary store for testing and fast iteration.

---

### 4. Server Layer (`src/server/`)

**`InferenceRouter`** runs as the main FastAPI application. On startup, it:
1. Launches `num_workers` worker subprocesses on ports `[worker_base_port, ...]`
2. Initializes an `httpx.AsyncClient` for async HTTP dispatch
3. Starts a background health check loop (polls `/health` every N seconds)
4. Applies the configured `RecoveryStrategy` to every incoming request

**`WorkerApp`** runs as a child process (spawned via `multiprocessing.Process`). Each worker:
1. Initializes its backend on startup
2. Serves `/infer` (blocking) and `/stream` (SSE) endpoints
3. Maintains a `FaultState` object that can be armed via `/admin/fault`
4. On the next request after fault is armed, fires the fault (sleep/raise/kill)

**`FaultState`** is a per-worker mutable object that stores the armed fault type and parameters. It fires exactly once per arming (or continuously for delay faults).

---

### 5. Metrics Layer (`src/utils/metrics.py`)

**`RequestMetrics`** captures timestamped events per request:
- `timestamp_start`, `timestamp_first_token`, `timestamp_end`
- `timestamp_recovery_start`, `timestamp_recovery_end`
- `tokens_committed_before_failure`, `tokens_recomputed`
- `success`, `num_retries`, `error_message`

Derived properties compute: `latency_s`, `time_to_first_token_s`, `recovery_time_s`, `recomputation_fraction`.

**`MetricsSink`** appends serialized `RequestMetrics` to a JSONL log and maintains an in-memory list for CSV export and aggregate computation.

**`compute_aggregate`** takes a list of `RequestMetrics` and returns `AggregateStats` with P50/P95/P99 latency, success rate, mean recovery time, mean recomputation fraction, and throughput.

---

## Fault Model

Five fault types are modeled:

| Fault | Mechanism | Simulated via |
|---|---|---|
| `none` | No fault | Baseline |
| `delay` | Artificial sleep before generation | `asyncio.sleep(delay_ms / 1000)` |
| `hang` | Long sleep that exceeds timeout | `asyncio.sleep(hang_s)` + raise TimeoutError |
| `kill` | SIGKILL to worker process | `os.kill(os.getpid(), SIGKILL)` |
| `graceful_shutdown` | SIGTERM to worker process | `os.kill(os.getpid(), SIGTERM)` |

Faults are armed via `/admin/fault` on the worker and fire on the next inference call.

---

## Request Lifecycle (Token-Commit-Resume)

```
T=0.000  Router receives POST /generate
T=0.001  Router creates RequestMetrics, RecoveryContext
T=0.001  Strategy: initialize DiskTokenStore checkpoint
T=0.002  Strategy: dispatch to worker via HTTP
T=0.012  Worker: apply_fault_if_active() → no fault
T=0.012  Worker: backend.generate(request) begins
T=0.012  Backend: sleep(base_latency_ms) → TTFT
T=0.062  Backend: generate tokens 0..47 (48 tokens)
[FAULT INJECTED: Worker SIGKILL at token 48]
T=0.520  Router: httpx receives ConnectError
T=0.520  Strategy: catches exception
T=0.520  Strategy: load checkpoint → tokens[0..47] (if streamed)
T=0.521  Strategy: sleep(backoff=0.5s)
T=1.021  Strategy: build resume_request(resume_from_tokens=[0..47])
T=1.021  Strategy: dispatch to worker-1 (round-robin)
T=1.022  Worker-1: backend.generate(resume_request)
T=1.022  Backend: yield tokens[0..47] instantly (from resume prefix)
T=1.022  Backend: generate tokens 48..127 (80 new tokens)
T=2.622  Response complete
T=2.622  Strategy: delete checkpoint
T=2.622  Router: return GenerateResponse
T=2.622  Router: record metrics (latency=2.622s, recovery=1.101s, recomputed=80)
```

Without Token-Commit-Resume (retry-from-scratch), the system would regenerate all 128 tokens instead of 80, adding `(48 / tokens_per_second)` to recovery time.

---

## Concurrency Model

The router uses `asyncio` for concurrent request handling. Workers are OS processes (bypassing Python GIL). The health check loop runs as an `asyncio.Task`. Request dispatch uses `httpx.AsyncClient` with connection pooling.

Worker selection (round-robin default) is O(n) in the number of healthy workers. The `least_loaded` strategy tracks `current_load` (incremented at dispatch start, decremented at completion).

---

## Token Checkpoint Granularity

The `checkpoint_every_n_tokens` parameter controls the tradeoff between:
- **Checkpoint I/O overhead**: smaller N → more disk writes → higher latency
- **Recovery granularity**: smaller N → fewer tokens recomputed after fault

With `checkpoint_every_n=10` and `tokens_per_second=50`:
- I/O overhead: ~0.2ms/token × (1 write per 10 tokens) = 0.02ms/token amortized
- Max tokens lost in worst case: 10 tokens = 200ms of wasted work

In Tier-2 (stub), KV-cache state is preserved alongside tokens, eliminating all recomputation overhead but requiring model-level integration.

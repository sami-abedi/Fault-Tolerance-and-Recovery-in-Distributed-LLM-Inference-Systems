"""
Smoke test for CrashSafe.

Verifies the core system works end-to-end WITHOUT requiring a running server.
Tests direct backend and recovery strategy invocation.

Usage:
    python scripts/smoke_test.py            # quick validation
    python scripts/smoke_test.py --verbose  # detailed output
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
INFO = "\033[94m→\033[0m"


def print_result(label: str, passed: bool, detail: str = "") -> None:
    symbol = PASS if passed else FAIL
    msg = f"  {symbol} {label}"
    if detail:
        msg += f": {detail}"
    print(msg)


async def run_smoke_tests(verbose: bool = False) -> int:
    """
    Run all smoke tests.

    Returns:
        Number of failed tests (0 = all passed).
    """
    failures = 0
    start = time.perf_counter()

    print("\n" + "="*60)
    print("  CrashSafe Smoke Test Suite")
    print("="*60 + "\n")

    # ----------------------------------------------------------------
    # 1. Config loading
    # ----------------------------------------------------------------
    print(f"{INFO} [1/7] Configuration loading")
    try:
        from src.config import load_config
        cfg = load_config()
        assert cfg.backend.type in ("mock", "transformers", "vllm")
        assert cfg.recovery.strategy in ("fail_stop", "retry_from_scratch", "token_commit_resume")
        print_result("Config loads successfully", True, f"backend={cfg.backend.type}")
    except Exception as e:
        print_result("Config loads successfully", False, str(e))
        failures += 1

    # ----------------------------------------------------------------
    # 2. Mock backend
    # ----------------------------------------------------------------
    print(f"\n{INFO} [2/7] Mock backend")
    try:
        from src.backends.mock_backend import MockBackend
        from src.backends.base import InferenceRequest

        backend = MockBackend({"tokens_per_second": 500.0, "base_latency_ms": 1.0})
        await backend.load()
        assert await backend.health_check()

        req = InferenceRequest(
            request_id="smoke-001",
            prompt="Fault tolerance in distributed systems",
            max_new_tokens=15,
        )
        response = await backend.generate(req)

        assert response.completion_tokens == 15
        assert len(response.generated_text) > 0
        print_result(
            "MockBackend generate()",
            True,
            f"{response.completion_tokens} tokens, text='{response.generated_text[:30]}...'"
        )

        # Determinism check
        r2 = await backend.generate(req)
        assert [t.token_id for t in response.tokens] == [t.token_id for t in r2.tokens]
        print_result("Deterministic generation", True)

    except Exception as e:
        print_result("MockBackend", False, str(e))
        failures += 1
        if verbose:
            import traceback
            traceback.print_exc()

    # ----------------------------------------------------------------
    # 3. Token-resume correctness
    # ----------------------------------------------------------------
    print(f"\n{INFO} [3/7] Token-Commit-Resume correctness")
    try:
        from src.backends.mock_backend import MockBackend, _deterministic_token_id
        from src.backends.base import InferenceRequest

        backend = MockBackend({"tokens_per_second": 500.0, "base_latency_ms": 1.0})
        await backend.load()

        prompt = "Test resume correctness"
        max_tokens = 20
        k = 8  # resume from position 8

        req_full = InferenceRequest("resume-full", prompt, max_new_tokens=max_tokens)
        full_resp = await backend.generate(req_full)
        full_ids = [t.token_id for t in full_resp.tokens]

        committed = full_ids[:k]
        req_resume = InferenceRequest(
            "resume-partial",
            prompt,
            max_new_tokens=max_tokens,
            resume_from_tokens=committed,
        )
        resume_resp = await backend.generate(req_resume)
        resume_ids = [t.token_id for t in resume_resp.tokens]

        # Suffix from position k should match
        assert resume_ids[k:] == full_ids[k:], "Resume suffix must match full generation"
        print_result(
            "Resume suffix matches full generation",
            True,
            f"resumed from pos {k}, suffix length {max_tokens - k}"
        )
    except Exception as e:
        print_result("Token-resume correctness", False, str(e))
        failures += 1

    # ----------------------------------------------------------------
    # 4. Fail-stop strategy
    # ----------------------------------------------------------------
    print(f"\n{INFO} [4/7] FailStop recovery strategy")
    try:
        from src.backends.base import InferenceRequest
        from src.recovery.fail_stop import FailStopStrategy
        from src.recovery.base import RecoveryContext
        from src.backends.base import InferenceResponse, Token

        strategy = FailStopStrategy()
        req = InferenceRequest("fs-test", "test", max_new_tokens=10)

        async def good_fn(r):
            return InferenceResponse(r.request_id, tokens=[], generated_text="ok",
                                     completion_tokens=10)

        async def bad_fn(r):
            raise RuntimeError("simulated crash")

        ctx = RecoveryContext(req, ["worker-0"])

        r_good = await strategy.execute(ctx, good_fn)
        assert r_good.success
        assert r_good.num_retries == 0
        print_result("FailStop success case", True)

        r_bad = await strategy.execute(ctx, bad_fn)
        assert not r_bad.success
        assert r_bad.num_retries == 0
        print_result("FailStop failure case (no retry)", True)

    except Exception as e:
        print_result("FailStop strategy", False, str(e))
        failures += 1

    # ----------------------------------------------------------------
    # 5. Retry strategy
    # ----------------------------------------------------------------
    print(f"\n{INFO} [5/7] RetryFromScratch strategy")
    try:
        from src.backends.base import InferenceRequest, InferenceResponse
        from src.recovery.retry import RetryFromScratchStrategy
        from src.recovery.base import RecoveryContext

        strategy = RetryFromScratchStrategy(max_retries=2, backoff_base_s=0.01)
        req = InferenceRequest("retry-test", "test", max_new_tokens=10)
        ctx = RecoveryContext(req, ["worker-0"])

        call_count = 0
        async def succeed_on_third(r):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient failure")
            return InferenceResponse(r.request_id, generated_text="success",
                                     completion_tokens=10, tokens=[])

        result = await strategy.execute(ctx, succeed_on_third)
        assert result.success
        assert result.num_retries == 2
        print_result("Retry succeeds on 3rd attempt", True, f"{call_count} calls made")

    except Exception as e:
        print_result("Retry strategy", False, str(e))
        failures += 1

    # ----------------------------------------------------------------
    # 6. Token-Commit-Resume strategy with MemoryStore
    # ----------------------------------------------------------------
    print(f"\n{INFO} [6/7] TokenCommitResume strategy")
    try:
        from src.backends.base import InferenceRequest, InferenceResponse, Token
        from src.recovery.token_resume import TokenCommitResumeStrategy
        from src.recovery.base import RecoveryContext
        from src.storage.token_store import MemoryTokenStore

        store = MemoryTokenStore()
        strategy = TokenCommitResumeStrategy(
            token_store=store,
            checkpoint_every_n=5,
            max_retries=2,
            backoff_base_s=0.01,
        )
        req = InferenceRequest("tcr-test", "test checkpoint", max_new_tokens=20)
        ctx = RecoveryContext(req, ["worker-0"])

        call_count = 0
        async def fail_then_succeed(r):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first attempt crash")
            tokens = [Token(i, f"t{i}", position=i) for i in range(20)]
            return InferenceResponse(r.request_id, tokens=tokens,
                                     generated_text="recovered", completion_tokens=20)

        result = await strategy.execute(ctx, fail_then_succeed)
        assert result.success
        assert result.num_retries >= 1
        assert not await store.exists(req.request_id)  # checkpoint cleaned up
        print_result(
            "TokenCommitResume recovers from fault",
            True,
            f"{result.num_retries} retries, {result.tokens_recomputed} tokens recomputed"
        )

    except Exception as e:
        print_result("TokenCommitResume strategy", False, str(e))
        failures += 1

    # ----------------------------------------------------------------
    # 7. Metrics collection
    # ----------------------------------------------------------------
    print(f"\n{INFO} [7/7] Metrics collection")
    try:
        import tempfile
        from src.utils.metrics import MetricsSink, RequestMetrics, compute_aggregate
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = MetricsSink(f"{tmpdir}/metrics.jsonl")
            records = []
            for i in range(10):
                r = RequestMetrics(f"m-{i}", fault_type="none", recovery_strategy="fail_stop")
                r.timestamp_start = time.time() - (0.5 + i * 0.1)
                r.timestamp_end = time.time()
                r.timestamp_first_token = r.timestamp_start + 0.05
                r.success = (i % 3 != 0)
                r.total_tokens_generated = 20
                sink.record(r)
                records.append(r)

            agg = compute_aggregate(records)
            assert agg.total_requests == 10
            assert 0 < agg.success_rate <= 1
            assert agg.p95_latency_s >= agg.p50_latency_s

        print_result(
            "Metrics collected and aggregated",
            True,
            f"success_rate={agg.success_rate:.0%} p50={agg.p50_latency_s*1000:.0f}ms"
        )

    except Exception as e:
        print_result("Metrics collection", False, str(e))
        failures += 1

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    elapsed = time.perf_counter() - start
    print("\n" + "="*60)
    total_tests = 7
    passed = total_tests - failures
    if failures == 0:
        print(f"  {PASS} All {total_tests} smoke tests passed in {elapsed:.2f}s")
    else:
        print(f"  {FAIL} {failures}/{total_tests} tests FAILED in {elapsed:.2f}s")
    print("="*60 + "\n")

    return failures


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="CrashSafe smoke test")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    failures = asyncio.run(run_smoke_tests(verbose=args.verbose))
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()

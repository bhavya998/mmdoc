"""Load test for mmdoc API — concurrent requests, p50/p99 latency, throughput.

Run: uv run python scripts/load_test.py
Requires: server running (uv run mmdoc serve)
"""

from __future__ import annotations

import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import requests

BASE_URL = "http://localhost:8000"
CONCURRENT_USERS = 10
TOTAL_REQUESTS = 50


def health_check() -> None:
    """Verify server is up."""
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        r.raise_for_status()
        print(f"Server healthy: {r.json()}")
    except Exception as e:
        print(f"Server not reachable at {BASE_URL}: {e}")
        sys.exit(1)


def single_request() -> float:
    """Make a single /health request, return latency in ms."""
    t0 = time.time()
    requests.get(f"{BASE_URL}/health", timeout=10)
    return (time.time() - t0) * 1000


def run_load_test() -> None:
    """Run concurrent load test and report stats."""
    print(f"\nLoad test: {CONCURRENT_USERS} users, {TOTAL_REQUESTS} requests")
    print("-" * 50)

    latencies: list[float] = []
    errors = 0

    with ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as pool:
        futures = [pool.submit(single_request) for _ in range(TOTAL_REQUESTS)]
        for i, future in enumerate(as_completed(futures)):
            try:
                lat = future.result()
                latencies.append(lat)
                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/{TOTAL_REQUESTS} done...")
            except Exception:
                errors += 1

    latencies.sort()
    p50 = statistics.median(latencies)
    p99 = latencies[int(len(latencies) * 0.99)]
    avg = statistics.mean(latencies)
    rps = len(latencies) / max(sum(latencies) / 1000 / CONCURRENT_USERS, 1)

    print("\n" + "=" * 50)
    print("LOAD TEST RESULTS")
    print("=" * 50)
    print(f"  Total requests:  {len(latencies)}")
    print(f"  Errors:          {errors}")
    print(f"  Avg latency:     {avg:.1f}ms")
    print(f"  p50 latency:     {p50:.1f}ms")
    print(f"  p99 latency:     {p99:.1f}ms")
    print(f"  Throughput:      {rps:.1f} req/s")
    print("=" * 50)


if __name__ == "__main__":
    health_check()
    run_load_test()

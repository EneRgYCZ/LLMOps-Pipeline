#!/usr/bin/env python3
"""HPA scaling load test for the chat-app multi-replica deployment."""

import argparse
import asyncio
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_HOST = "http://localhost:8000"
DEFAULT_WORKERS = 6
DEFAULT_DURATION = 300
REQUEST_TIMEOUT = 120.0
SUMMARY_INTERVAL = 15  # seconds
PROMPT = "What is 2+2? Answer in one word."

REPO_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(REPO_ROOT / ".env")


def load_model_name() -> str:
    return os.getenv("OLLAMA_MODEL", "ministral-3:3b-instruct-2512-q4_K_M")


# ---------------------------------------------------------------------------
# kubectl helper
# ---------------------------------------------------------------------------

_KUBECTL_CONTEXT: str | None = None


def get_ollama_replicas() -> str:
    cmd = ["kubectl"]
    if _KUBECTL_CONTEXT:
        cmd += ["--context", _KUBECTL_CONTEXT]
    cmd += [
        "get", "deployment", "ollama",
        "-n", "llmops",
        "-o", "jsonpath={.status.readyReplicas}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return "N/A"
        val = result.stdout.strip()
        return val if val else "0"
    except Exception:
        return "N/A"


# ---------------------------------------------------------------------------
# Shared state (written by workers, read by reporter)
# ---------------------------------------------------------------------------

class Stats:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.rows: list[dict] = []
        self.error_count: int = 0
        self.peak_replicas: int = 0
        self.first_scale_time: float | None = None  # seconds from start


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

async def worker(
    worker_id: int,
    host: str,
    model: str,
    end_time: float,
    stats: Stats,
    csv_writer_lock: asyncio.Lock,
    csv_writer: csv.DictWriter,
    start_time: float,
) -> None:
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        while time.monotonic() < end_time:
            req_start = time.monotonic()
            req_ts = datetime.utcnow().isoformat()
            status_code = 0
            success = False
            try:
                resp = await client.post(url, json=payload)
                status_code = resp.status_code
                if status_code == 200:
                    data = resp.json()
                    # Ollama response: {"message": {"content": "..."}}
                    content = (
                        data.get("message", {}).get("content", "")
                        or data.get("response", "")
                    )
                    success = bool(content.strip())
            except Exception:
                pass

            duration = time.monotonic() - req_start

            if not success:
                async with stats.lock:
                    stats.error_count += 1

            row = {
                "timestamp": req_ts,
                "worker_id": worker_id,
                "duration_seconds": round(duration, 4),
                "status_code": status_code,
                "success": success,
            }

            async with stats.lock:
                stats.rows.append(row)

            async with csv_writer_lock:
                csv_writer.writerow(row)


# ---------------------------------------------------------------------------
# Periodic reporter
# ---------------------------------------------------------------------------

async def reporter(
    stats: Stats,
    start_time: float,
    end_time: float,
) -> None:
    last_count = 0
    while time.monotonic() < end_time:
        await asyncio.sleep(SUMMARY_INTERVAL)
        elapsed = time.monotonic() - start_time

        async with stats.lock:
            rows_snapshot = list(stats.rows)
            errors = stats.error_count

        total = len(rows_snapshot)
        new_requests = total - last_count
        last_count = total

        replica_str = get_ollama_replicas()
        try:
            replicas = int(replica_str)
            async with stats.lock:
                if replicas > stats.peak_replicas:
                    stats.peak_replicas = replicas
                if replicas > 1 and stats.first_scale_time is None:
                    stats.first_scale_time = elapsed
        except ValueError:
            pass

        durations = [r["duration_seconds"] for r in rows_snapshot if r["success"]]
        mean_lat = sum(durations) / len(durations) if durations else 0.0
        sorted_d = sorted(durations)
        p50 = sorted_d[int(len(sorted_d) * 0.50)] if sorted_d else 0.0
        p95 = sorted_d[int(len(sorted_d) * 0.95)] if sorted_d else 0.0
        rps = total / elapsed if elapsed > 0 else 0.0

        print(
            f"[{elapsed:6.0f}s] reqs={total:5d}  rps={rps:.2f}"
            f"  mean={mean_lat:.2f}s  p50={p50:.2f}s  p95={p95:.2f}s"
            f"  errors={errors}  ollama_replicas={replica_str}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HPA scaling load test")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--output", default=None, help="Override CSV output path")
    parser.add_argument("--kubectl-context", default=None, help="kubectl context for replica count queries")
    return parser.parse_args()


async def main() -> None:
    global _KUBECTL_CONTEXT
    args = parse_args()
    _KUBECTL_CONTEXT = args.kubectl_context
    model = load_model_name()

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(args.output) if args.output else results_dir / f"load_test_{ts}.csv"

    print(f"Load test config:")
    print(f"  host           : {args.host}")
    print(f"  workers        : {args.workers}")
    print(f"  duration       : {args.duration}s")
    print(f"  model          : {model}")
    print(f"  output         : {csv_path}")
    print(f"  timeout        : {REQUEST_TIMEOUT}s/request")
    print(f"  kubectl-context: {args.kubectl_context or '(default)'}")
    print()

    stats = Stats()
    csv_writer_lock = asyncio.Lock()
    fieldnames = ["timestamp", "worker_id", "duration_seconds", "status_code", "success"]

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        csvfile.flush()

        start_time = time.monotonic()
        end_time = start_time + args.duration

        tasks = [
            asyncio.create_task(
                worker(i, args.host, model, end_time, stats, csv_writer_lock, writer, start_time)
            )
            for i in range(args.workers)
        ]
        tasks.append(
            asyncio.create_task(reporter(stats, start_time, end_time))
        )

        await asyncio.gather(*tasks)

    # Final summary
    elapsed = time.monotonic() - start_time
    async with stats.lock:
        rows = list(stats.rows)
        errors = stats.error_count
        peak = stats.peak_replicas
        first_scale = stats.first_scale_time

    total = len(rows)
    durations = [r["duration_seconds"] for r in rows if r["success"]]
    mean_lat = sum(durations) / len(durations) if durations else 0.0
    sorted_d = sorted(durations)
    p50 = sorted_d[int(len(sorted_d) * 0.50)] if sorted_d else 0.0
    p95 = sorted_d[int(len(sorted_d) * 0.95)] if sorted_d else 0.0
    rps = total / elapsed if elapsed > 0 else 0.0

    print()
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  total duration     : {elapsed:.1f}s")
    print(f"  total requests     : {total}")
    print(f"  requests/sec       : {rps:.2f}")
    print(f"  mean latency       : {mean_lat:.2f}s")
    print(f"  p50 latency        : {p50:.2f}s")
    print(f"  p95 latency        : {p95:.2f}s")
    print(f"  errors             : {errors}")
    print(f"  peak ollama replicas: {peak if peak > 0 else 'N/A (kubectl unreachable or wrong context)'}")
    if first_scale is not None:
        print(f"  time-to-first-scale: {first_scale:.1f}s")
    else:
        print(f"  time-to-first-scale: no scale event observed")
    print(f"  CSV output         : {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

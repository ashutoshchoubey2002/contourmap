"""
Admission control and result caching.

Analysis is CPU-bound for several seconds per request. Four unbounded
concurrent requests on a container that shares a host with three others will
not fail cleanly — they will all slow down together until the load balancer's
health check times out and the worker is marked dead, taking its in-flight work
with it.

So the worker accepts a bounded number of analyses, refuses the rest with 429
and a Retry-After, and reports its own load honestly. Refusing fast is a better
failure than degrading silently: the load balancer can route a 429 elsewhere,
but it cannot route a request that is merely slow.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time

from app import config

# --------------------------------------------------------------- admission

_sem = threading.BoundedSemaphore(config.MAX_CONCURRENT_JOBS)
_state_lock = threading.Lock()
_inflight = 0
_accepted = 0
_rejected = 0
_durations: list[float] = []


class Busy(RuntimeError):
    """Worker is at capacity. Maps to 429."""

    def __init__(self, retry_after: int):
        super().__init__("worker at capacity")
        self.retry_after = retry_after


class Slot:
    """Context manager holding one analysis slot. Non-blocking: if no slot is
    free the request is refused rather than queued, because a queued request
    still occupies a connection and still eventually times out."""

    def __enter__(self):
        global _inflight, _accepted, _rejected
        if not _sem.acquire(blocking=False):
            with _state_lock:
                _rejected += 1
            raise Busy(config.RETRY_AFTER_S)
        with _state_lock:
            _inflight += 1
            _accepted += 1
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc):
        global _inflight
        elapsed = time.perf_counter() - self.started
        with _state_lock:
            _inflight -= 1
            _durations.append(elapsed)
            if len(_durations) > 200:
                del _durations[:-200]
        _sem.release()
        return False


def _cpu_percent() -> float:
    """CPU of this process, not the machine.

    All four systems are containers on one host, so a machine-level figure is
    identical everywhere and tells the load balancer nothing about which worker
    is actually busy. This was the exact failure in the messaging lab.
    """
    try:
        with open("/proc/self/stat") as fh:
            parts = fh.read().split()
        ticks = float(parts[13]) + float(parts[14])
        hz = os.sysconf("SC_CLK_TCK")
        now = time.time()
        prev = getattr(_cpu_percent, "_prev", None)
        _cpu_percent._prev = (ticks / hz, now)
        if prev is None:
            return 0.0
        dt = now - prev[1]
        return min(100.0, 100.0 * ((ticks / hz) - prev[0]) / dt) if dt > 0 else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def load_report() -> dict:
    with _state_lock:
        d = sorted(_durations)
        p50 = d[len(d) // 2] if d else 0.0
        p95 = d[int(len(d) * 0.95)] if len(d) >= 20 else (d[-1] if d else 0.0)
        return {
            "inflight": _inflight,
            "capacity": config.MAX_CONCURRENT_JOBS,
            "accepted": _accepted,
            "rejected": _rejected,
            "cpu_percent": round(_cpu_percent(), 1),
            # Fraction of capacity in use, for a threshold-based balancer.
            "load": round(100.0 * _inflight / config.MAX_CONCURRENT_JOBS, 1),
            "p50_seconds": round(p50, 3),
            "p95_seconds": round(p95, 3),
        }


# --------------------------------------------------------------- result cache
#
# Keyed on the rounded polygon plus the parameters that change the answer.
# Rounding to ~10 m means a repeated demo, or a user nudging the box by one
# pixel, is a hit. On the shared deployment this directory should be the same
# path on every worker, so a result computed on sys1 serves a later request
# landing on sys3.

_mem: dict[str, tuple[dict, float]] = {}
_mem_lock = threading.Lock()


def cache_key(ring, params: dict) -> str:
    pts = [[round(float(x), 4), round(float(y), 4)] for x, y in ring]
    payload = json.dumps(
        {"ring": pts, "p": {k: params[k] for k in sorted(params)}},
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _disk_path(key: str) -> str:
    return os.path.join(config.RESULT_CACHE_DIR, f"{key}.json")


def cache_get(key: str) -> dict | None:
    now = time.time()
    with _mem_lock:
        hit = _mem.get(key)
    if hit and now - hit[1] < config.RESULT_TTL_S:
        return hit[0]

    path = _disk_path(key)
    try:
        if now - os.path.getmtime(path) > config.RESULT_TTL_S:
            return None
        with open(path) as fh:
            value = json.load(fh)
    except (OSError, ValueError):
        return None

    with _mem_lock:
        _mem[key] = (value, now)
    return value


def cache_put(key: str, value: dict) -> None:
    now = time.time()
    with _mem_lock:
        _mem[key] = (value, now)
        if len(_mem) > config.RESULT_MEM_MAX:
            oldest = sorted(_mem.items(), key=lambda kv: kv[1][1])[: len(_mem) // 4]
            for k, _ in oldest:
                _mem.pop(k, None)

    path = _disk_path(key)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        os.makedirs(config.RESULT_CACHE_DIR, exist_ok=True)
        with open(tmp, "w") as fh:
            json.dump(value, fh, separators=(",", ":"))
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def cache_stats() -> dict:
    try:
        n = len([f for f in os.listdir(config.RESULT_CACHE_DIR) if f.endswith(".json")])
    except OSError:
        n = 0
    with _mem_lock:
        return {"memory": len(_mem), "disk": n}
"""
Annual rainfall for a location.

Sources are tried in order: a local annual-rainfall raster (WorldClim bio12 or
a precomputed IMD mean), NASA POWER climatology, then Open-Meteo ERA5. If all
fail, the caller's fallback is used and the reason is returned so the UI can
say why instead of silently presenting a default as a measurement.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from app import config

log = logging.getLogger(__name__)

_cache: dict[tuple[float, float], tuple[float, float, str]] = {}
_lock = threading.Lock()
_raster = None
_failed: dict[tuple[float, float], tuple[float, str]] = {}
_FAIL_TTL_S = 600   # after every source fails, don't retry for 10 minutes
POWER_URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"


def _key(lat: float, lon: float) -> tuple[float, float]:
    g = config.RAINFALL_GRID_DEG
    return (round(round(lat / g) * g, 4), round(round(lon / g) * g, 4))


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": config.TILE_USER_AGENT})
    with urllib.request.urlopen(req, timeout=config.RAINFALL_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


class _NotConfigured(Exception):
    """An optional source that isn't set up; skipped without comment."""


def _fetch_local(lat: float, lon: float) -> float:
    """Sample the local annual-rainfall GeoTIFF (mm/year) at the exact point."""
    global _raster
    path = getattr(config, "RAINFALL_RASTER", None)
    if not path or not os.path.exists(path):
        raise _NotConfigured(path)
    import rasterio
    if _raster is None:
        _raster = rasterio.open(path)
    b = _raster.bounds
    if not (b.left <= lon <= b.right and b.bottom <= lat <= b.top):
        raise ValueError("point outside raster")
    v = float(next(_raster.sample([(lon, lat)]))[0])
    if (_raster.nodata is not None and v == _raster.nodata) or v <= 0:
        raise ValueError("nodata at point")
    return v


def _fetch_power(lat: float, lon: float) -> float:
    """NASA POWER long-term climatology. ANN is reported in mm/day."""
    q = urllib.parse.urlencode({"parameters": "PRECTOTCORR", "community": "AG",
                                "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
                                "format": "JSON"})
    d = _get_json(f"{POWER_URL}?{q}")
    ann = d["properties"]["parameter"]["PRECTOTCORR"]["ANN"]
    units = (d.get("parameters", {}).get("PRECTOTCORR", {}).get("units") or "mm/day").lower()
    if ann is None or ann < 0:          # POWER uses -999 for missing
        raise ValueError("POWER returned no value")
    return ann * 365.25 if "day" in units else float(ann)


def _fetch_era5(lat: float, lon: float) -> float:
    """Mean annual total precipitation from the Open-Meteo ERA5 archive."""
    q = urllib.parse.urlencode({"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
                                "start_date": config.RAINFALL_START,
                                "end_date": config.RAINFALL_END,
                                "daily": "precipitation_sum", "timezone": "UTC"})
    d = _get_json(f"{config.RAINFALL_URL}?{q}")
    values = [v for v in (d.get("daily", {}).get("precipitation_sum") or []) if v is not None]
    if not values:
        raise ValueError("no precipitation data returned")
    return sum(values) / max(len(values) / 365.25, 0.5)


SOURCES = (("local", _fetch_local, False),   # (name, fn, use snapped grid key)
           ("power", _fetch_power, True),
           ("era5", _fetch_era5, True))


def _describe(e: Exception) -> str:
    if isinstance(e, urllib.error.HTTPError):
        try:
            body = e.read().decode("utf-8", "replace")[:160]
        except Exception:  # noqa: BLE001
            body = ""
        return f"HTTP {e.code} {body}".strip()
    return f"{type(e).__name__}: {e}"


def annual_rainfall_mm(lat: float, lon: float, fallback: float | None = None
                       ) -> tuple[float, str, str | None]:
    """Return (millimetres, source, reason).

    source is 'local', 'power', 'era5', 'cache:<src>' or 'fallback'; reason is
    None unless every source failed.
    """
    fallback = config.RAINFALL_DEFAULT_MM if fallback is None else fallback
    k = _key(lat, lon)

    with _lock:
        hit = _cache.get(k)
    if hit and (time.time() - hit[1]) < config.RAINFALL_TTL_S:
        return hit[0], f"cache:{hit[2]}", None

    with _lock:
        miss = _failed.get(k)
    if miss and (time.time() - miss[0]) < _FAIL_TTL_S:
        return float(fallback), "fallback", miss[1]

    errors = []
    for name, fn, snapped in SOURCES:
        try:
            mm = fn(*(k if snapped else (lat, lon)))
        except _NotConfigured:
            continue
        except Exception as e:  # noqa: BLE001 - network, parsing, anything
            msg = _describe(e)
            log.warning("rainfall source %s failed at %s: %s", name, k, msg)
            errors.append(f"{name}: {msg}")
            continue
        with _lock:
            _cache[k] = (mm, time.time(), name)
            _failed.pop(k, None)
        return mm, name, None

    reason = "; ".join(errors)
    with _lock:
        _failed[k] = (time.time(), reason)
    return float(fallback), "fallback", reason


def runoff_coefficient(slope_pct: float, override: float | None = None) -> float:
    """Rational-method C, interpolated on mean catchment slope."""
    if override is not None:
        return float(override)
    if slope_pct <= 1.0:
        return 0.15
    if slope_pct <= 3.0:
        return 0.20
    if slope_pct <= 5.0:
        return 0.30
    if slope_pct <= 10.0:
        return 0.40
    return 0.50

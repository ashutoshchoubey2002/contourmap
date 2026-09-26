"""
Annual rainfall for a location.

The contour phase took rainfall as a request parameter and documented that as a
limitation. This resolves it: mean annual precipitation is read from the
Open-Meteo archive (ERA5, no API key) at the selection centroid.

Two deliberate choices. Lookups are cached on a 0.25 degree grid, because ERA5
itself is a 0.25 degree product and asking for two points 10 km apart returns
the same cell — caching finer would be caching noise. And any failure falls
back to the caller's supplied value rather than raising, because a blocked
firewall at demo time should cost accuracy, not the whole result.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
import urllib.request

from app import config

_cache: dict[tuple[float, float], tuple[float, float]] = {}
_lock = threading.Lock()


def _key(lat: float, lon: float) -> tuple[float, float]:
    g = config.RAINFALL_GRID_DEG
    return (round(lat / g) * g, round(lon / g) * g)


def _fetch(lat: float, lon: float) -> float:
    """Mean annual total precipitation over the configured reference period."""
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": config.RAINFALL_START,
        "end_date": config.RAINFALL_END,
        "daily": "precipitation_sum",
        "timezone": "UTC",
    }
    url = config.RAINFALL_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": config.TILE_USER_AGENT})
    with urllib.request.urlopen(req, timeout=config.RAINFALL_TIMEOUT_S) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    daily = data.get("daily", {}).get("precipitation_sum") or []
    values = [v for v in daily if v is not None]
    if not values:
        raise ValueError("no precipitation data returned")

    total = sum(values)
    years = max(len(values) / 365.25, 0.5)
    return total / years


def annual_rainfall_mm(lat: float, lon: float, fallback: float | None = None
                       ) -> tuple[float, str]:
    """Return (millimetres, source). Source is one of 'era5', 'cache',
    'fallback' — surfaced in the response so a demo cannot silently present a
    default as a measurement."""
    fallback = config.RAINFALL_DEFAULT_MM if fallback is None else fallback
    k = _key(lat, lon)

    with _lock:
        hit = _cache.get(k)
    if hit and (time.time() - hit[1]) < config.RAINFALL_TTL_S:
        return hit[0], "cache"

    try:
        mm = _fetch(k[0], k[1])
    except Exception:  # noqa: BLE001 - network, parsing, anything
        return float(fallback), "fallback"

    with _lock:
        _cache[k] = (mm, time.time())
    return mm, "era5"


def runoff_coefficient(slope_pct: float, override: float | None = None) -> float:
    """Rational-method C, interpolated on mean catchment slope.

    Values follow the usual agricultural-catchment range. Slope is the only
    terrain variable available here — land use and soil would improve this and
    are the obvious next input.
    """
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
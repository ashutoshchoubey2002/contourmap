"""
Elevation from terrain-RGB tiles.

The contour path builds a DEM by interpolating placemarks. There is no such
file when the user drags a box on a map, so elevation is read from public
terrain tiles instead: SRTM/NED merged, terrarium-encoded PNGs, ~30 m at the
equator, no API key.

This module is deliberately ignorant of projections and hydrology. It returns
a mosaic in Web Mercator tile space; app/dem/grid.py turns that into the
metric grid the pipeline expects.
"""

from __future__ import annotations

import io
import math
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from PIL import Image

from app import config

TILE_PX = 256


# --------------------------------------------------------------- tile indexing

def deg2num(lat: float, lon: float, z: float) -> tuple[float, float]:
    """Slippy-map tile coordinates. Fractional, so it also locates a point
    inside a tile."""
    # numpy throughout: grid.py calls this with whole coordinate arrays.
    lat = np.clip(lat, -85.05112878, 85.05112878)
    rad = np.radians(lat)
    n = 2.0 ** z
    x = (np.asarray(lon) + 180.0) / 360.0 * n
    y = (1.0 - np.arcsinh(np.tan(rad)) / np.pi) / 2.0 * n
    return x, y


def num2deg(x: float, y: float, z: float) -> tuple[float, float]:
    """Inverse of deg2num."""
    n = 2.0 ** z
    lon = np.asarray(x) / n * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * np.asarray(y) / n))))
    return lat, lon


def ground_resolution(lat: float, z: int) -> float:
    """Metres per pixel at a given latitude and zoom."""
    return 156543.03392804097 * math.cos(math.radians(lat)) / (2.0 ** z)


def choose_zoom(bounds: tuple[float, float, float, float],
                target_cells: int = None,
                zmin: int = None,
                zmax: int = None) -> int:
    """Pick the zoom whose pixel grid is closest to target_cells on the long
    side, without exceeding it, and never finer than the source data.

    This is the whole defence against a user dragging a box over half a state.
    Cost stays roughly constant in pixels regardless of the area selected; what
    degrades is resolution, which is the right thing to trade. The caller is
    told what it got so it can report the cell size honestly.
    """
    target_cells = target_cells or config.TILE_TARGET_CELLS
    zmin = zmin if zmin is not None else config.TILE_ZOOM_MIN
    zmax = zmax if zmax is not None else config.TILE_ZOOM_MAX

    west, south, east, north = bounds
    lat_mid = (south + north) / 2.0

    # Long side of the selection in metres, via a local equirectangular
    # approximation. Good to a fraction of a percent at these extents.
    m_per_deg_lat = 111132.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat_mid))
    width_m = abs(east - west) * m_per_deg_lon
    height_m = abs(north - south) * m_per_deg_lat
    long_side_m = max(width_m, height_m, 1.0)

    # Never sample finer than the underlying data. The tiles are built from
    # SRTM/NED at about 30 m; asking for a finer zoom repeats each source
    # value across a block of pixels, and those blocks are exactly flat.
    # Flat cells have no steepest descent, so D8 cannot route across them and
    # the drainage network fragments. Measured on a uniform slope: 8.9 m
    # cells from 35 m source gave a 2.3 ha maximum catchment and no stream
    # network at all; 35.6 m cells on the same terrain gave 1,017 ha.
    for z in range(zmax, zmin - 1, -1):
        if ground_resolution(lat_mid, z) >= config.TILE_MIN_CELL_M:
            zmax = z
            break

    best = zmin
    for z in range(zmin, zmax + 1):
        cells = long_side_m / ground_resolution(lat_mid, z)
        if cells <= target_cells:
            best = z
        else:
            break
    return best


def tile_range(bounds: tuple[float, float, float, float], z: int
               ) -> tuple[int, int, int, int]:
    """Inclusive tile index range covering bounds at zoom z.

    Returns (x0, y0, x1, y1). Note y is inverted relative to latitude: north
    is a smaller y.
    """
    west, south, east, north = bounds
    x0f, y0f = deg2num(north, west, z)
    x1f, y1f = deg2num(south, east, z)
    n = 2 ** z
    x0 = max(0, int(math.floor(x0f)))
    y0 = max(0, int(math.floor(y0f)))
    x1 = min(n - 1, int(math.floor(x1f)))
    y1 = min(n - 1, int(math.floor(y1f)))
    return x0, y0, x1, y1


def iter_tiles(x0: int, y0: int, x1: int, y1: int) -> Iterator[tuple[int, int]]:
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            yield x, y


# --------------------------------------------------------------- disk cache
#
# Terrain tiles are immutable: (z, x, y) always denotes the same ground. So the
# cache never needs invalidating, and on the four-system deployment it should
# live on a path shared by every worker. Hit rate climbs steeply once a few
# demos have run over the same district.

_cache_lock = threading.Lock()


def _cache_path(z: int, x: int, y: int) -> str:
    return os.path.join(config.TILE_CACHE_DIR, str(z), str(x), f"{y}.png")


def _cache_read(z: int, x: int, y: int) -> bytes | None:
    path = _cache_path(z, x, y)
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        return None


def _cache_write(z: int, x: int, y: int, blob: bytes) -> None:
    path = _cache_path(z, x, y)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with _cache_lock:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "wb") as fh:
            fh.write(blob)
        # Atomic, so a worker that dies mid-write cannot leave a truncated PNG
        # for another worker to read.
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def cache_stats() -> dict:
    """Tile count and bytes on disk. Reported by /health so the cache can be
    shown to be working during the demo rather than merely asserted."""
    count = 0
    size = 0
    for root, _dirs, files in os.walk(config.TILE_CACHE_DIR):
        for f in files:
            if f.endswith(".png"):
                count += 1
                try:
                    size += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    return {"tiles": count, "bytes": size}


# --------------------------------------------------------------- fetching

class TileError(RuntimeError):
    """Tile could not be obtained. Carries the tile so the caller can decide
    whether a hole is tolerable."""

    def __init__(self, z: int, x: int, y: int, reason: str):
        super().__init__(f"tile {z}/{x}/{y}: {reason}")
        self.z, self.x, self.y = z, x, y
        self.reason = reason


def _fetch_one(z: int, x: int, y: int) -> tuple[bytes, bool]:
    """Return (png_bytes, from_cache). Retries with backoff on transient
    failures; raises TileError once retries are spent."""
    blob = _cache_read(z, x, y)
    if blob is not None:
        return blob, True

    url = config.TILE_URL.format(z=z, x=x, y=y)
    last = "unknown"
    for attempt in range(config.TILE_RETRIES):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": config.TILE_USER_AGENT}
            )
            with urllib.request.urlopen(req, timeout=config.TILE_TIMEOUT_S) as resp:
                blob = resp.read()
            if not blob:
                raise urllib.error.URLError("empty body")
            _cache_write(z, x, y, blob)
            return blob, False
        except urllib.error.HTTPError as exc:
            # 404 means no coverage — ocean, or past the edge of the dataset.
            # Retrying will not help, and it is not an error condition.
            if exc.code == 404:
                raise TileError(z, x, y, "no coverage") from exc
            last = f"http {exc.code}"
        except Exception as exc:  # noqa: BLE001 - network is diverse
            last = type(exc).__name__
        time.sleep(config.TILE_BACKOFF_S * (2 ** attempt))
    raise TileError(z, x, y, last)


def _decode_terrarium(blob: bytes) -> np.ndarray:
    """Terrarium encoding: elevation = R*256 + G + B/256 - 32768, in metres.

    Returns float32 (256, 256). The offset means sea level is not zero in the
    raw channels, so a naive read looks like a 32 km high plateau.
    """
    with Image.open(io.BytesIO(blob)) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return arr[:, :, 0] * 256.0 + arr[:, :, 1] + arr[:, :, 2] / 256.0 - 32768.0


# --------------------------------------------------------------- mosaic

@dataclass
class TileMosaic:
    """A rectangular block of elevation in tile-pixel space.

    elevation : (H, W) float32, metres, NaN where no tile was available
    z         : zoom the mosaic was built at
    px0, py0  : global pixel offset of elevation[0, 0] at this zoom
    """

    elevation: np.ndarray
    z: int
    px0: int
    py0: int
    tiles_total: int
    tiles_cached: int
    tiles_missing: int
    fetch_seconds: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation.shape

    def pixel_to_lonlat(self, col: float, row: float) -> tuple[float, float]:
        """Mosaic pixel -> (lon, lat). Cell centres are at col+0.5."""
        gx = (self.px0 + col) / TILE_PX
        gy = (self.py0 + row) / TILE_PX
        lat, lon = num2deg(gx, gy, self.z)
        return lon, lat

    def lonlat_to_pixel(self, lon: float, lat: float) -> tuple[float, float]:
        """(lon, lat) -> mosaic pixel. Inverse of pixel_to_lonlat."""
        gx, gy = deg2num(lat, lon, self.z)
        return gx * TILE_PX - self.px0, gy * TILE_PX - self.py0

    def bounds(self) -> tuple[float, float, float, float]:
        """(west, south, east, north) of the mosaic's outer edge."""
        h, w = self.elevation.shape
        west, north = self.pixel_to_lonlat(0, 0)
        east, south = self.pixel_to_lonlat(w, h)
        return west, south, east, north

    def mean_cell_size_m(self) -> float:
        _west, south, _east, north = self.bounds()
        return ground_resolution((south + north) / 2.0, self.z)


def fetch_mosaic(bounds: tuple[float, float, float, float],
                 z: int | None = None,
                 allow_missing: bool = True) -> TileMosaic:
    """Fetch every tile covering bounds and stitch them into one array.

    bounds is (west, south, east, north) in WGS-84 degrees and should already
    include the buffer — flow routing needs terrain outside the selection to
    resolve a catchment that drains in from upslope.
    """
    if z is None:
        z = choose_zoom(bounds)

    x0, y0, x1, y1 = tile_range(bounds, z)
    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    total = nx * ny

    if total > config.TILE_MAX_TILES:
        raise ValueError(
            f"selection needs {total} tiles at zoom {z}, limit is "
            f"{config.TILE_MAX_TILES}; reduce the area"
        )

    mosaic = np.full((ny * TILE_PX, nx * TILE_PX), np.nan, dtype=np.float32)
    cached = 0
    missing = 0
    started = time.perf_counter()

    def job(t: tuple[int, int]):
        # The exception is returned rather than raised: pool.map re-raises at
        # the consuming loop, so one uncovered tile would abort the mosaic.
        x, y = t
        try:
            return t, _fetch_one(z, x, y)
        except TileError as exc:
            return t, exc

    # Modest pool. The bottleneck is the remote server, and opening thirty
    # sockets at once against a public tile service invites rate limiting.
    workers = min(config.TILE_CONCURRENCY, max(1, total))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (x, y), result in pool.map(job, iter_tiles(x0, y0, x1, y1)):
            row = (y - y0) * TILE_PX
            col = (x - x0) * TILE_PX
            if isinstance(result, TileError):
                if not allow_missing:
                    raise result
                missing += 1
                continue
            blob, from_cache = result
            cached += 1 if from_cache else 0
            mosaic[row:row + TILE_PX, col:col + TILE_PX] = _decode_terrarium(blob)

    elapsed = time.perf_counter() - started

    if missing == total:
        raise TileError(z, x0, y0, "no elevation data for this area")

    return TileMosaic(
        elevation=mosaic,
        z=z,
        px0=x0 * TILE_PX,
        py0=y0 * TILE_PX,
        tiles_total=total,
        tiles_cached=cached,
        tiles_missing=missing,
        fetch_seconds=elapsed,
    )
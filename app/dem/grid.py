"""
Tile mosaic -> metric grid.

Flow routing assumes square cells of known size in metres. A tile mosaic is in
Web Mercator pixel space, where cell size varies with latitude and the axes are
not metric. This module resamples the mosaic onto a local UTM grid so that
app/terrain/flow.py can run on it unchanged.

It also carries the selection mask: flow routing runs on the buffered grid, but
candidate pond sites must lie inside the polygon the user actually drew.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from pyproj import CRS, Transformer

from app import config
from app.dem.tiles import TileMosaic, fetch_mosaic, choose_zoom


# --------------------------------------------------------------- geometry

def polygon_bounds(ring: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    """(west, south, east, north) of a [(lon, lat), ...] ring."""
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    return min(lons), min(lats), max(lons), max(lats)


def buffer_bounds(bounds, frac: float | None = None):
    """Expand bounds outward by a fraction of each side.

    Flow arriving at a pond originates upslope, often outside the drawn box.
    Routing on the unbuffered selection guarantees a truncated catchment, which
    is exactly the limitation the contour phase could not fix.
    """
    frac = config.AREA_BUFFER_FRAC if frac is None else frac
    west, south, east, north = bounds
    dx = (east - west) * frac
    dy = (north - south) * frac
    return (max(west - dx, -180.0), max(south - dy, -85.0),
            min(east + dx, 180.0), min(north + dy, 85.0))


def polygon_area_km2(ring: list[tuple[float, float]]) -> float:
    """Shoelace on a local equirectangular projection. Accurate to well under
    a percent at village-to-district extents, and needs no projection setup."""
    lat0 = sum(p[1] for p in ring) / len(ring)
    kx = 111.320 * math.cos(math.radians(lat0))
    ky = 110.574
    pts = [(p[0] * kx, p[1] * ky) for p in ring]
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def points_in_polygon(xs: np.ndarray, ys: np.ndarray,
                      ring: list[tuple[float, float]]) -> np.ndarray:
    """Vectorised even-odd ray casting. Avoids a shapely dependency in the hot
    path; xs/ys are arrays of identical shape."""
    inside = np.zeros(xs.shape, dtype=bool)
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if y1 == y2:
            continue
        crosses = (ys > min(y1, y2)) & (ys <= max(y1, y2))
        x_at = x1 + (ys - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (xs < x_at)
    return inside


def utm_crs_for(lon: float, lat: float) -> CRS:
    """Local UTM zone. Metric, near-conformal, and square enough that a single
    cell size is valid across a selection of this size."""
    zone = int((lon + 180.0) / 6.0) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


# --------------------------------------------------------------- DEM object

@dataclass
class AreaDEM:
    """Metric elevation grid, north-up, square cells.

    Field names match what app/terrain/flow.py already consumes from the
    contour path. If your terrain/dem.py names them differently, rename here
    only — nothing downstream should need editing.
    """

    elevation: np.ndarray          # (rows, cols) float32, metres, NaN = nodata
    cell_size: float               # metres, square
    x0: float                      # UTM easting of centre of column 0
    y0: float                      # UTM northing of centre of row 0 (north edge)
    crs: CRS                       # projected CRS of x0/y0
    selection_mask: np.ndarray     # True where inside the user's polygon
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.elevation.shape

    @property
    def nodata_mask(self) -> np.ndarray:
        return ~np.isfinite(self.elevation)

    def cell_area_m2(self) -> float:
        return self.cell_size * self.cell_size

    def rowcol_to_xy(self, row, col):
        """Grid index -> UTM. Row increases southward, so northing decreases."""
        return self.x0 + np.asarray(col) * self.cell_size, \
               self.y0 - np.asarray(row) * self.cell_size

    def rowcol_to_lonlat(self, row, col):
        """Grid index -> (lon, lat). Used for every GeoJSON coordinate."""
        x, y = self.rowcol_to_xy(row, col)
        tr = Transformer.from_crs(self.crs, CRS.from_epsg(4326), always_xy=True)
        lon, lat = tr.transform(x, y)
        return lon, lat


# --------------------------------------------------------------- build

def _bilinear(src: np.ndarray, cols: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Bilinear sample of src at fractional (col, row). Out-of-range and any
    sample touching NaN yields NaN, so nodata never bleeds inward."""
    h, w = src.shape
    c0 = np.floor(cols).astype(np.int64)
    r0 = np.floor(rows).astype(np.int64)
    valid = (c0 >= 0) & (r0 >= 0) & (c0 < w - 1) & (r0 < h - 1)

    c0c = np.clip(c0, 0, w - 2)
    r0c = np.clip(r0, 0, h - 2)
    fc = (cols - c0c)[..., None] if False else (cols - c0c)
    fr = rows - r0c

    v00 = src[r0c, c0c]
    v01 = src[r0c, c0c + 1]
    v10 = src[r0c + 1, c0c]
    v11 = src[r0c + 1, c0c + 1]

    top = v00 * (1 - fc) + v01 * fc
    bot = v10 * (1 - fc) + v11 * fc
    out = top * (1 - fr) + bot * fr
    return np.where(valid, out, np.nan).astype(np.float32)


def build_from_mosaic(mosaic: TileMosaic,
                      ring: list[tuple[float, float]],
                      clip_bounds: tuple[float, float, float, float] | None = None,
                      cell_size: float | None = None) -> AreaDEM:
    """Resample a mosaic onto a UTM grid.

    clip_bounds restricts the grid to the buffered selection. Without it the
    grid would cover whole tiles, which can be several times the area asked
    for — pure cost, since flow routing is O(n log n) in cells.
    """
    west, south, east, north = clip_bounds or mosaic.bounds()
    lon_c, lat_c = (west + east) / 2.0, (south + north) / 2.0
    crs = utm_crs_for(lon_c, lat_c)

    fwd = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    inv = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)

    # Project the corners and take the enclosing axis-aligned box. UTM is not
    # axis-aligned with lat/lon, so sampling the four corners alone would clip
    # the edges; the midpoints of each side bound the curvature.
    bx = [west, east, west, east, lon_c, lon_c, west, east]
    by = [south, south, north, north, south, north, lat_c, lat_c]
    xs, ys = fwd.transform(bx, by)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    if cell_size is None:
        cell_size = round(mosaic.mean_cell_size_m(), 2)

    cols = int(math.floor((xmax - xmin) / cell_size))
    rows = int(math.floor((ymax - ymin) / cell_size))
    cols = max(cols, 2)
    rows = max(rows, 2)

    # Cell centres, half a cell in from the box edge.
    gx = xmin + (np.arange(cols) + 0.5) * cell_size
    gy = ymax - (np.arange(rows) + 0.5) * cell_size
    gxx, gyy = np.meshgrid(gx, gy)

    glon, glat = inv.transform(gxx.ravel(), gyy.ravel())
    glon = np.asarray(glon).reshape(rows, cols)
    glat = np.asarray(glat).reshape(rows, cols)

    pc, pr = mosaic.lonlat_to_pixel(glon, glat)
    elevation = _bilinear(mosaic.elevation, pc, pr)

    mask = points_in_polygon(glon, glat, ring)

    return AreaDEM(
        elevation=elevation,
        cell_size=float(cell_size),
        x0=float(gx[0]),
        y0=float(gy[0]),
        crs=crs,
        selection_mask=mask,
        meta={
            "source": "terrarium",
            "zoom": mosaic.z,
            "cell_size_m": float(cell_size),
            "grid": [rows, cols],
            "tiles_total": mosaic.tiles_total,
            "tiles_cached": mosaic.tiles_cached,
            "tiles_missing": mosaic.tiles_missing,
            "tile_fetch_seconds": round(mosaic.fetch_seconds, 3),
            "epsg": crs.to_epsg(),
        },
    )


def build_from_polygon(ring: list[tuple[float, float]],
                       buffer_frac: float | None = None) -> AreaDEM:
    """Entry point for the area path: polygon -> ready-to-route DEM.

    Raises ValueError if the selection exceeds the configured area cap; the
    caller should surface that as 400 rather than attempting the analysis.
    """
    if len(ring) < 3:
        raise ValueError("polygon needs at least three vertices")

    area_km2 = polygon_area_km2(ring)
    if area_km2 > config.AREA_MAX_KM2:
        raise ValueError(
            f"selection is {area_km2:.1f} km2, limit is {config.AREA_MAX_KM2} km2"
        )
    if area_km2 < config.AREA_MIN_KM2:
        raise ValueError(
            f"selection is {area_km2:.3f} km2, too small to resolve terrain"
        )

    bounds = buffer_bounds(polygon_bounds(ring), buffer_frac)
    mosaic = fetch_mosaic(bounds, choose_zoom(bounds))
    dem = build_from_mosaic(mosaic, ring, clip_bounds=bounds)
    dem.meta["selection_area_km2"] = round(area_km2, 3)
    dem.meta["buffer_frac"] = config.AREA_BUFFER_FRAC if buffer_frac is None else buffer_frac
    return dem
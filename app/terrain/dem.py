"""
ContourSet -> DEM grid.

Contour lines sirf kuch jagah par height batati hain. Beech ki
values interpolation se nikalte hain, aur regular grid par rakh
dete hain. Uske baad pura kaam numbers par hota hai, geometry par
nahi -- isliye aage ke saare steps simple ho jaate hain.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from pyproj import Transformer


@dataclass
class DEM:
    z: np.ndarray            # (ny, nx), NaN = data ke bahar
    cell: float              # meters
    x_min: float
    y_min: float
    epsg: int
    meta: dict

    @property
    def shape(self):
        return self.z.shape

    @property
    def cell_area(self):
        return self.cell * self.cell

    def rc_to_xy(self, r, c):
        return self.x_min + c * self.cell, self.y_min + r * self.cell

    def to_lonlat(self):
        return Transformer.from_crs(f"EPSG:{self.epsg}", "EPSG:4326",
                                    always_xy=True)


# ---------------------------------------------------------------

def utm_epsg(lon, lat):
    """Data ke apne centroid se UTM zone nikalo -- hard-coded nahi."""
    zone = int((lon + 180) / 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def build(cs, settings):
    """ContourSet -> DEM."""
    lon, lat, z = cs.lon, cs.lat, cs.elev

    # bahut zyada points hon to subsample -- warna triangulation slow
    if len(lon) > settings.max_points:
        step = int(math.ceil(len(lon) / settings.max_points))
        lon, lat, z = lon[::step], lat[::step], z[::step]
        subsampled = step
    else:
        subsampled = 1

    epsg = utm_epsg(float(lon.mean()), float(lat.mean()))
    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = tf.transform(lon, lat)
    x, y = np.asarray(x), np.asarray(y)

    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    w, h = x1 - x0, y1 - y0

    cell = float(np.clip(max(w, h) / settings.target_cells,
                         settings.cell_min_m, settings.cell_max_m))
    nx = int(math.ceil(w / cell)) + 1
    ny = int(math.ceil(h / cell)) + 1

    # duplicate (x,y) hata do -- Delaunay ko pasand nahi
    key = np.round(np.column_stack([x, y]), 2)
    _, idx = np.unique(key, axis=0, return_index=True)
    idx.sort()

    interp = LinearNDInterpolator(np.column_stack([x[idx], y[idx]]), z[idx])
    gx = x0 + np.arange(nx) * cell
    gy = y0 + np.arange(ny) * cell
    grid = interp(*np.meshgrid(gx, gy))

    valid = int((~np.isnan(grid)).sum())
    meta = {
        "cell_size_m": round(cell, 2),
        "grid_rows": ny,
        "grid_cols": nx,
        "extent_m": {"width": round(w, 1), "height": round(h, 1)},
        "extent_km2": round(w * h / 1e6, 3),
        "valid_cells": valid,
        "total_cells": int(grid.size),
        "coverage_pct": round(100 * valid / grid.size, 1),
        "points_used": int(len(idx)),
        "subsample_step": subsampled,
        "dem_min_m": float(np.nanmin(grid)),
        "dem_max_m": float(np.nanmax(grid)),
    }
    return DEM(grid, cell, x0, y0, epsg, meta)

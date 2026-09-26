"""
Site selection, water accounting and GeoJSON export.

Two quantities get reported and they are not the same thing, which is the usual
source of confusion in rainwater-harvesting output:

  annual yield     what the catchment delivers in a year, P x A x C
  storage capacity what a pond at that site can actually hold at one time

A catchment of a few hundred hectares delivers far more water annually than any
village pond can store. Reporting only the first overstates the structure;
reporting only the second hides the resource. Both are returned, along with the
number of times the pond could be refilled.
"""

from __future__ import annotations

import math

import numpy as np
from skimage.measure import find_contours

from app import config
from app.area import hydro


# --------------------------------------------------------------- scoring

def _norm(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Percentile-clipped min-max, so one extreme cell cannot flatten the rest."""
    out = np.zeros_like(a, dtype=np.float32)
    vals = a[mask]
    if vals.size == 0:
        return out
    lo, hi = np.percentile(vals, [2, 98])
    if hi - lo < 1e-9:
        return out
    out[mask] = np.clip((vals - lo) / (hi - lo), 0.0, 1.0)
    return out


def score_cells(flow: hydro.FlowResult, dem, streams: np.ndarray,
                excluded: np.ndarray | None = None) -> np.ndarray:
    """Suitability per cell, 0 to 1.

    Four terms, weights in config:

      runoff      more contributing area means more water intercepted
      flatness    gentle ground means less excavation per cubic metre stored
      concavity   naturally concave ground already holds water
      depth       existing depression depth, free storage

    Candidates are limited to channel cells: a pond off-channel intercepts
    little regardless of how good its other scores look.
    """
    valid = np.isfinite(flow.filled) & dem.selection_mask & streams
    if excluded is not None:
        # Applied before normalisation, not after scoring: excluded cells must
        # not influence the percentile range the remaining cells are scaled
        # against, or one built-up valley floor skews every other score.
        valid &= ~excluded

    runoff = _norm(np.log1p(flow.accumulation), valid)
    slope = hydro.slope_percent(flow.filled, dem.cell_size)
    flatness = 1.0 - _norm(slope, valid)

    # Concavity: filled surface minus a local mean. Negative where the ground
    # dishes inward.
    k = 5
    pad = np.pad(np.nan_to_num(flow.filled, nan=0.0), k, mode="edge")
    local = np.zeros_like(flow.filled)
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            local += pad[k + dr: k + dr + flow.filled.shape[0],
                         k + dc: k + dc + flow.filled.shape[1]]
    local /= (2 * k + 1) ** 2
    concavity = _norm(local - np.nan_to_num(flow.filled, nan=0.0), valid)

    depth = _norm(flow.fill_depth, valid)

    s = (config.W_RUNOFF * runoff
         + config.W_FLATNESS * flatness
         + config.W_CONCAVITY * concavity
         + config.W_DEPTH * depth)
    return np.where(valid, s, 0.0).astype(np.float32)


def pick_sites(score: np.ndarray, dem, n: int | None = None) -> list[dict]:
    """Top-scoring cells with non-maximum suppression.

    Without suppression every candidate lands within a few cells of the best
    one, which is a single recommendation reported five times. The exclusion
    radius comes from config in metres so it is independent of resolution.
    """
    n = config.MAX_CANDIDATES if n is None else n
    radius_cells = max(1, int(round(config.CANDIDATE_MIN_SEP_M / dem.cell_size)))

    work = score.copy()
    rows, cols = work.shape
    out = []
    for _ in range(n):
        idx = int(np.argmax(work))
        r, c = divmod(idx, cols)
        if work[r, c] <= 0:
            break
        out.append({"row": r, "col": c, "score": float(round(work[r, c], 4))})
        r0, r1 = max(0, r - radius_cells), min(rows, r + radius_cells + 1)
        c0, c1 = max(0, c - radius_cells), min(cols, c + radius_cells + 1)
        work[r0:r1, c0:c1] = 0.0
    return out


# --------------------------------------------------------------- water

def annual_yield_m3(catchment_ha: float, rainfall_mm: float, runoff_c: float) -> float:
    """P x A x C. Area in hectares, rainfall in millimetres."""
    return catchment_ha * 10_000.0 * (rainfall_mm / 1000.0) * runoff_c


def pond_geometry(depth_m: float | None = None,
                  side_slope: float | None = None,
                  top_area_m2: float | None = None) -> dict:
    """Capacity of a truncated-pyramid excavation.

    A pond is not a box. With side slopes of 1:z the bed is smaller than the
    surface, and treating it as a box overstates capacity by roughly 20-30% at
    village scale. Prismoidal formula is used instead.
    """
    depth = config.POND_DEPTH_M if depth_m is None else depth_m
    z = config.POND_SIDE_SLOPE if side_slope is None else side_slope
    top = config.POND_TOP_AREA_M2 if top_area_m2 is None else top_area_m2

    side = math.sqrt(top)                      # assume square footprint
    bed = max(side - 2.0 * z * depth, 1.0)     # inward batter on both sides
    a_top, a_bed = side * side, bed * bed
    a_mid = ((side + bed) / 2.0) ** 2
    volume = (depth / 6.0) * (a_top + 4.0 * a_mid + a_bed)   # prismoidal
    return {
        "depth_m": round(depth, 2),
        "side_slope": z,
        "top_area_m2": round(a_top, 1),
        "bed_area_m2": round(a_bed, 1),
        "capacity_m3": round(volume, 1),
        "excavation_m3": round(volume, 1),
    }


def natural_storage_m3(flow: hydro.FlowResult, dem, mask: np.ndarray) -> float:
    """Volume held by the existing depression, from the fill depths.

    This is storage that costs nothing to create. Where it is large the site is
    a natural tank and excavation can be reduced.
    """
    return float(np.nansum(flow.fill_depth[mask]) * dem.cell_area_m2())


def water_summary(catchment_ha: float, rainfall_mm: float, runoff_c: float,
                  natural_m3: float, pond: dict) -> dict:
    yield_m3 = annual_yield_m3(catchment_ha, rainfall_mm, runoff_c)
    capacity = pond["capacity_m3"] + natural_m3
    return {
        "annual_yield_m3": round(yield_m3, 1),
        "storage_capacity_m3": round(capacity, 1),
        "natural_storage_m3": round(natural_m3, 1),
        "excavated_capacity_m3": pond["capacity_m3"],
        # How many times the catchment could refill the structure in a year.
        # Above about 3 the pond is the binding constraint, not the catchment.
        "refills_per_year": round(yield_m3 / capacity, 2) if capacity > 0 else None,
        "rainfall_mm": round(rainfall_mm, 1),
        "runoff_coefficient": runoff_c,
    }


# --------------------------------------------------------------- geojson

def mask_to_polygon(mask: np.ndarray, dem, simplify_cells: float = 1.5) -> dict | None:
    """Largest connected boundary of a boolean mask, as a GeoJSON polygon.

    find_contours traces the half-value isoline of the padded mask, which puts
    the boundary on cell edges rather than through cell centres.
    """
    padded = np.pad(mask.astype(float), 1)
    contours = find_contours(padded, 0.5)
    if not contours:
        return None
    contour = max(contours, key=len)
    contour = _simplify(contour, simplify_cells)

    rows = contour[:, 0] - 1.0
    cols = contour[:, 1] - 1.0
    lon, lat = dem.rowcol_to_lonlat(rows, cols)
    ring = [[float(a), float(b)] for a, b in zip(np.atleast_1d(lon), np.atleast_1d(lat))]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return {"type": "Polygon", "coordinates": [ring]}


def _simplify(points: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker, iterative. A 400-cell boundary becomes a few dozen
    vertices, which keeps the response small enough to draw smoothly."""
    if len(points) < 3:
        return points
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        p, q = points[i], points[j]
        seg = q - p
        L = math.hypot(*seg)
        rel = points[i + 1:j] - p
        if L < 1e-12:
            d = np.hypot(rel[:, 0], rel[:, 1])
        else:
            # 2-D cross product written out. np.cross no longer accepts
            # 2-vectors in current numpy, and it was only ever the z
            # component of a 3-D cross with the z terms zeroed.
            d = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / L
        k = int(np.argmax(d))
        if d[k] > tol:
            k += i + 1
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    return points[keep]


def streams_to_geojson(streams: np.ndarray, flow: hydro.FlowResult, dem,
                       max_segments: int = 1500) -> dict:
    """Channel network as MultiLineString, one segment per stream cell.

    Capped, because a dense network on a large grid produces a payload the
    browser will not draw at interactive speed. The highest-accumulation
    segments are kept, which are the ones that read as the main channels.
    """
    cols = dem.shape[1]
    idx = np.flatnonzero(streams.ravel())
    if idx.size == 0:
        return {"type": "MultiLineString", "coordinates": []}

    acc = flow.accumulation.ravel()[idx]
    if idx.size > max_segments:
        idx = idx[np.argsort(-acc)[:max_segments]]

    segments = []
    down = flow.downstream
    for i in idx:
        d = down[i]
        if d < 0:
            continue
        r0, c0 = divmod(int(i), cols)
        r1, c1 = divmod(int(d), cols)
        lon0, lat0 = dem.rowcol_to_lonlat(r0, c0)
        lon1, lat1 = dem.rowcol_to_lonlat(r1, c1)
        segments.append([[float(lon0), float(lat0)], [float(lon1), float(lat1)]])
    return {"type": "MultiLineString", "coordinates": segments}


def touches_edge(mask: np.ndarray) -> bool:
    """True if the catchment reaches the analysed grid edge, in which case the
    reported area is a lower bound."""
    return bool(mask[0, :].any() or mask[-1, :].any()
                or mask[:, 0].any() or mask[:, -1].any())
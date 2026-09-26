"""
Catchment delineation.

Pour point (pond ki jagah) se ULTA chalte hain: "mujhme paani kaun
de raha hai?" Phir unse wahi poochte hain. Jitne cells mile, wahi
catchment.
"""

import numpy as np

from ..terrain.flow import DR, DC


def delineate(fdir, r0, c0):
    """
    Upstream BFS.

    Padosi (r+DR[k], c+DC[k]) mujhme behta hai agar uski flow
    direction bilkul ulti ho, yaani (k+4)%8. Poora delineation
    isi ek shart par tika hai.
    """
    ny, nx = fdir.shape
    mask = np.zeros((ny, nx), bool)
    mask[r0, c0] = True
    stack = [(r0, c0)]

    while stack:
        r, c = stack.pop()
        for k in range(8):
            rr, cc = r + int(DR[k]), c + int(DC[k])
            if rr < 0 or cc < 0 or rr >= ny or cc >= nx or mask[rr, cc]:
                continue
            if fdir[rr, cc] == (k + 4) % 8:
                mask[rr, cc] = True
                stack.append((rr, cc))
    return mask


def touches_edge(mask, filled=None):
    """
    Catchment data ki boundary tak pahunch raha hai?

    Sirf array ka kinara dekhna kaafi nahi. DEM ke chaaron taraf NaN
    ka border hota hai (contours ke convex hull ke bahar), to
    catchment DATA ki boundary chhoo sakta hai bina ARRAY ki boundary
    chhue -- aur tab check kabhi trigger hi nahi hota.

    Isliye do cheezein dekhte hain:
      1. array ka kinara
      2. koi NaN cell se lagav

    True aane ka matlab: asli catchment map se bahar tak jaata hai,
    aur hamara area sirf LOWER BOUND hai. Ye batana imaandari hai.
    """
    if (mask[0, :].any() or mask[-1, :].any()
            or mask[:, 0].any() or mask[:, -1].any()):
        return True

    if filled is None:
        return False

    nodata = np.isnan(filled)
    if not nodata.any():
        return False

    from scipy.ndimage import binary_dilation
    grown = binary_dilation(mask, structure=np.ones((3, 3), bool))
    return bool((grown & nodata).any())


# ---------------------------------------------------------------
# Polygon nikalna
# ---------------------------------------------------------------

def _perp_dist(P, i0, i1):
    """
    Line (P[i0] -> P[i1]) se har beech ke point ki doori.
    numpy 2.0 mein 2D np.cross hata diya gaya, isliye cross khud:
        a x b = ax*by - ay*bx
    """
    p0, p1 = P[i0], P[i1]
    seg = p1 - p0
    L = float(np.hypot(seg[0], seg[1]))
    v = P[i0:i1 + 1] - p0
    if L < 1e-9:
        return np.hypot(v[:, 0], v[:, 1])
    return np.abs(seg[0] * v[:, 1] - seg[1] * v[:, 0]) / L


def simplify(points, eps):
    """Douglas-Peucker (iterative). Polygon ke points kam karo."""
    P = np.asarray(points, float)
    n = len(P)
    if n < 3:
        return [tuple(p) for p in P]

    keep = np.zeros(n, bool)
    keep[0] = keep[n - 1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        d = _perp_dist(P, i0, i1)
        j = int(np.argmax(d))
        if d[j] > eps:
            j += i0
            keep[j] = True
            stack.append((i0, j))
            stack.append((j, i1))
    return [tuple(p) for p in P[keep]]


def to_geojson(mask, dem, simplify_m=None):
    """Catchment mask -> lon/lat GeoJSON Polygon."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if simplify_m is None:
        simplify_m = dem.cell * 1.5

    fig = plt.figure()
    cs = plt.contour(mask.astype(float), levels=[0.5])
    segs = [s for s in cs.allsegs[0] if len(s) > 3]
    plt.close(fig)
    if not segs:
        return None

    ring = max(segs, key=len)                # sabse bada = outer boundary
    pts = simplify([(float(p[0]), float(p[1])) for p in ring],
                   simplify_m / dem.cell)

    tf = dem.to_lonlat()
    coords = []
    for c, r in pts:
        x, y = dem.rc_to_xy(r, c)
        lon, lat = tf.transform(x, y)
        coords.append([round(float(lon), 7), round(float(lat), 7)])

    if len(coords) < 4:
        return None
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return {"type": "Polygon", "coordinates": [coords]}


# ---------------------------------------------------------------

def stats(mask, dem, flow):
    """Catchment ke numbers."""
    n = int(mask.sum())
    area = n * dem.cell_area
    z = flow.filled[mask]
    return {
        "cells": n,
        "area_m2": round(area, 1),
        "area_ha": round(area / 1e4, 2),
        "area_km2": round(area / 1e6, 4),
        "mean_slope_pct": round(float(np.nanmean(flow.slope[mask]) * 100), 2),
        "min_elev_m": round(float(np.nanmin(z)), 2),
        "max_elev_m": round(float(np.nanmax(z)), 2),
        "relief_m": round(float(np.nanmax(z) - np.nanmin(z)), 2),
        "truncated_by_map_edge": touches_edge(mask, flow.filled),
    }


def pond_sizing(area_m2, settings, rainfall_mm, mean_slope_pct,
                rainfall_source="request", rainfall_reason=None):
    """
    Water volumes for one site -- the same calculation the area path uses,
    so both routes report identical figures for identical terrain.

    Annual runoff (Rational method):   V = A x P x C
        A  catchment area (m2), P  annual rainfall (m), C  runoff coefficient
    Pond storage (prismoidal formula): V = d/6 x (A_top + 4 A_mid + A_bed)
        a square pond with 1:z side slopes, so the bed is smaller than the top
    Refills per year = annual runoff / storage. Above ~3 the pond, not the
    catchment, limits how much water is captured.
    """
    from app import rainfall as rain_mod
    from app.area import siting as area_siting

    c = rain_mod.runoff_coefficient(mean_slope_pct, settings.runoff_coefficient)
    runoff_m3 = area_m2 * (rainfall_mm / 1000.0) * c

    d = settings.pond_depth_m
    pond = area_siting.pond_geometry(depth_m=d)
    capacity = pond["capacity_m3"]

    # Surface a pond of this depth would need to hold the whole year's runoff
    # in one filling (box approximation, kept from phase 1 for comparison).
    full_top = runoff_m3 / d if d > 0 else 0.0

    return {
        # expected water volume that can be collected in a year
        "estimated_annual_runoff_m3": round(runoff_m3, 1),
        "rainfall_mm": round(float(rainfall_mm), 1),
        "rainfall_source": rainfall_source,
        "rainfall_reason": rainfall_reason,
        "runoff_coefficient": c,
        # the recommended pond and what it stores at once
        "suggested_side_m": round(float(np.sqrt(pond["top_area_m2"])), 1),
        "depth_m": pond["depth_m"],
        "side_slope": pond["side_slope"],
        "top_area_m2": pond["top_area_m2"],
        "bed_area_m2": pond["bed_area_m2"],
        "capacity_m3": capacity,
        "refills_per_year": round(runoff_m3 / capacity, 2) if capacity > 0 else None,
        # a pond sized to catch everything in one fill
        "full_capture_top_area_m2": round(full_top, 1),
        "full_capture_side_m": round(float(np.sqrt(max(full_top, 0.0))), 1),
        # phase-1 key names, kept so older clients keep working
        "assumed_rainfall_mm": round(float(rainfall_mm), 1),
        "recommended_depth_m": d,
        "required_top_area_m2": round(full_top, 1),
    }
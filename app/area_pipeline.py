"""
Area path orchestration.

Kept in its own module rather than folded into pipeline.py so that the working
contour path is not modified at all during this phase. The two share nothing
but config; once the area path is validated, pipeline.py can delegate to it.

Stage timings are collected throughout and returned. They are the evidence for
the performance claims in the report, and they make it obvious which stage to
attack when a large selection is slow.
"""

from __future__ import annotations

import time

import numpy as np

from app import config, limits, rainfall
from app.area import exclude, hydro, siting
from app.dem import grid as demgrid


class Timer:
    def __init__(self):
        self.marks: dict[str, float] = {}
        self._t = time.perf_counter()

    def mark(self, name: str):
        now = time.perf_counter()
        self.marks[name] = round(now - self._t, 3)
        self._t = now

    def total(self) -> float:
        return round(sum(self.marks.values()), 3)


def analyze_area(ring: list[tuple[float, float]],
                 rainfall_mm: float | None = None,
                 runoff_coefficient: float | None = None,
                 use_cache: bool = True) -> dict:
    """Polygon of (lon, lat) -> pond site, catchment and water volumes.

    Raises ValueError for anything the user can fix (too large, too small,
    degenerate) and limits.Busy when the worker is saturated.
    """
    params = {
        "rainfall_mm": rainfall_mm,
        "runoff_coefficient": runoff_coefficient,
        "version": config.RESULT_SCHEMA_VERSION,
    }
    key = limits.cache_key(ring, params)
    if use_cache:
        cached = limits.cache_get(key)
        if cached is not None:
            out = dict(cached)
            out["cached"] = True
            return out

    with limits.Slot():
        result = _run(ring, rainfall_mm, runoff_coefficient)

    result["cached"] = False
    limits.cache_put(key, result)
    return result


def _run(ring, rainfall_mm, runoff_c) -> dict:
    t = Timer()

    # 1. Elevation -------------------------------------------------------
    dem = demgrid.build_from_polygon(ring)
    t.mark("dem")

    # 2. Routing ---------------------------------------------------------
    flow = hydro.route(dem)
    t.mark("routing")

    cell_area = dem.cell_area_m2()
    streams = hydro.stream_mask(flow.accumulation, cell_area)
    if not (streams & dem.selection_mask).any():
        # Flat or featureless terrain: no channel crosses the selection, so
        # there is nothing meaningful to intercept. Say so rather than
        # returning the least-bad cell as if it were a recommendation.
        raise ValueError(
            "no drainage channel found inside the selection; the terrain is "
            "too flat or the area too small to site a pond"
        )

    # 3. Exclusions ------------------------------------------------------
    # Settlement, existing water and roads, plus ground too steep to dig.
    # Steep ground needs no network, so it applies even when Overpass is down.
    slope_all = hydro.slope_percent(flow.filled, dem.cell_size)
    veto = exclude.build(dem, demgrid.buffer_bounds(demgrid.polygon_bounds(ring)))
    blocked = veto.mask | exclude.steep_ground(slope_all)
    t.mark("exclusions")

    # 4. Siting ----------------------------------------------------------
    score = siting.score_cells(flow, dem, streams, excluded=blocked)
    candidates = siting.pick_sites(score, dem)
    if not candidates:
        raise ValueError(
            "no suitable pond site found: every channel inside the selection "
            "is settlement, existing water, road or ground too steep to dig"
            if veto.available else
            "no suitable pond site found inside the selection"
        )
    t.mark("siting")

    # 4. Catchment of the best candidate ---------------------------------
    best = candidates[0]
    cols = dem.shape[1]
    pour = best["row"] * cols + best["col"]
    catch_mask = hydro.delineate(flow.downstream, dem.shape, pour)
    catch_cells = int(catch_mask.sum())
    catchment_ha = catch_cells * cell_area / 10_000.0
    truncated = siting.touches_edge(catch_mask)
    t.mark("catchment")

    # 5. Water -----------------------------------------------------------
    lon_c, lat_c = dem.rowcol_to_lonlat(best["row"], best["col"])
    mean_slope = float(np.nanmean(slope_all[catch_mask]))

    mm, rain_source = rainfall.annual_rainfall_mm(float(lat_c), float(lon_c), rainfall_mm)
    if rainfall_mm is not None:
        mm, rain_source = float(rainfall_mm), "request"
    c = rainfall.runoff_coefficient(mean_slope, runoff_c)

    pond = siting.pond_geometry()
    natural = siting.natural_storage_m3(flow, dem, catch_mask & _disc(dem, best))
    water = siting.water_summary(catchment_ha, mm, c, natural, pond)
    t.mark("water")

    # 6. Geometry for the map --------------------------------------------
    catchment_geom = siting.mask_to_polygon(catch_mask, dem)
    stream_geom = siting.streams_to_geojson(streams, flow, dem)
    t.mark("geojson")

    elev = flow.filled[catch_mask]
    relief = float(np.nanmax(elev) - np.nanmin(elev)) if elev.size else 0.0

    # Alternatives carry their own catchment so the response shape matches
    # /analyzeContour exactly. Delineation is ~5 ms per site, so this costs
    # almost nothing and lets one renderer serve both paths.
    alternatives = []
    for rank, cand in enumerate(candidates[1:], start=2):
        lo, la = dem.rowcol_to_lonlat(cand["row"], cand["col"])
        m = hydro.delineate(flow.downstream, dem.shape, cand["row"] * cols + cand["col"])
        alternatives.append({
            "rank": rank,
            "score": cand["score"],
            "pond_location": {
                "lat": round(float(la), 6),
                "lon": round(float(lo), 6),
                "elevation_m": round(float(flow.filled[cand["row"], cand["col"]]), 2),
            },
            "catchment": {
                "area_ha": round(int(m.sum()) * cell_area / 10_000.0, 2),
                "truncated_by_map_edge": siting.touches_edge(m),
            },
        })
    t.mark("alternatives")

    return {
        "schema": config.RESULT_SCHEMA_VERSION,
        "mode": "area",
        "recommended_site": {
            "rank": 1,
            "score": best["score"],
            "pond_location": {
                "lat": round(float(lat_c), 6),
                "lon": round(float(lon_c), 6),
                "elevation_m": round(float(flow.filled[best["row"], best["col"]]), 2),
            },
            "catchment": {
                "area_ha": round(catchment_ha, 2),
                "area_km2": round(catchment_ha / 100.0, 4),
                "cells": catch_cells,
                "mean_slope_pct": round(mean_slope, 2),
                "relief_m": round(relief, 2),
                "truncated_by_map_edge": truncated,
                "geometry": catchment_geom,
            },
            "pond_sizing": {
                # Name kept from the contour path so existing consumers keep
                # working; the storage figures are additions.
                "estimated_annual_runoff_m3": water["annual_yield_m3"],
                "suggested_side_m": round(float(pond["top_area_m2"] ** 0.5), 1),
                "depth_m": pond["depth_m"],
                "side_slope": pond["side_slope"],
                "capacity_m3": water["storage_capacity_m3"],
                "excavated_capacity_m3": water["excavated_capacity_m3"],
                "natural_storage_m3": water["natural_storage_m3"],
                "refills_per_year": water["refills_per_year"],
                "rainfall_mm": water["rainfall_mm"],
                "runoff_coefficient": water["runoff_coefficient"],
                "rainfall_source": rain_source,
            },
        },
        "alternative_sites": alternatives,
        "streams": stream_geom,
        "source": {
            "area": {
                "selection_km2": dem.meta.get("selection_area_km2"),
                "buffer_fraction": dem.meta.get("buffer_frac"),
                "elevation_source": dem.meta.get("source"),
                "zoom": dem.meta.get("zoom"),
                "tiles_total": dem.meta.get("tiles_total"),
                "tiles_cached": dem.meta.get("tiles_cached"),
                "tiles_missing": dem.meta.get("tiles_missing"),
                "tile_fetch_sec": dem.meta.get("tile_fetch_seconds"),
            },
            "exclusions": veto.summary(),
        },
        "dem": {
            "cell_size_m": dem.meta.get("cell_size_m"),
            "grid": dem.meta.get("grid"),
            "epsg": dem.meta.get("epsg"),
            "cells_raised": flow.cells_raised,
            "max_fill_m": round(flow.max_fill, 2),
        },
        "timing": t.marks,
        "total_sec": t.total(),
    }


def _disc(dem, cand) -> np.ndarray:
    """Cells within the pond footprint radius of the chosen site, used to sum
    the natural depression storage actually available at the structure."""
    radius = max(1, int(round(
        (config.POND_TOP_AREA_M2 ** 0.5) / dem.cell_size)))
    rows, cols = dem.shape
    rr, cc = np.ogrid[:rows, :cols]
    return ((rr - cand["row"]) ** 2 + (cc - cand["col"]) ** 2) <= radius ** 2
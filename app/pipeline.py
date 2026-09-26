"""
Poora pipeline ek jagah.

    bytes -> ContourSet -> DEM -> FlowNet -> sites -> catchments -> JSON

Har stage alag module mein hai. Phase-2 mein koi ek stage badalni
ho (jaise KML ki jagah GeoTIFF) to sirf wahi module badlega.
"""

import time

from .config import DEFAULTS
from .parsers import kml
from .terrain import dem as dem_mod
from .terrain import flow as flow_mod
from .siting import score as score_mod
from .siting import catchment as catch_mod


def analyze(file_bytes, filename="", settings=None, include_geometry=True):
    s = settings or DEFAULTS
    t0 = time.time()
    timings = {}

    def tick(name):
        timings[name] = round(time.time() - t0 - sum(timings.values()), 2)

    # 1. parse
    cs = kml.parse(file_bytes, filename, s.level_gap_factor)
    tick("parse")

    # 2. DEM
    d = dem_mod.build(cs, s)
    tick("dem")

    # 3. flow
    f = flow_mod.analyze(d, s)
    tick("flow")

    # 4. site scoring
    sc, parts, mask = score_mod.score(d, f, s)
    sites = score_mod.pick_sites(sc, d, s)
    tick("scoring")

    if not sites:
        raise ValueError(
            "Koi suitable pond site nahi mila. "
            "min_catchment_ha kam karke dekho, ya map bahut chhota hai."
        )

    # 5. catchments
    tf = d.to_lonlat()

    # Rainfall is looked up once, at the best site, and used for every
    # candidate: they sit within a few km of each other, well inside one
    # 0.25 degree rainfall cell, and one lookup keeps a slow network from
    # costing five timeouts.
    if s.annual_rainfall_mm is not None:
        rain_mm, rain_src, rain_why = float(s.annual_rainfall_mm), "request", None
    else:
        from . import rainfall
        r0, c0 = sites[0]
        lon0, lat0 = tf.transform(*d.rc_to_xy(r0, c0))
        rain_mm, rain_src, rain_why = rainfall.annual_rainfall_mm(float(lat0), float(lon0))
    tick("rainfall")

    results = []
    for i, (r, c) in enumerate(sites):
        cm = catch_mod.delineate(f.fdir, r, c)
        st = catch_mod.stats(cm, d, f)
        x, y = d.rc_to_xy(r, c)
        lon, lat = tf.transform(x, y)

        entry = {
            "rank": i + 1,
            "score": round(float(sc[r, c]), 4),
            "score_components": {k: round(float(v[r, c]), 3)
                                 for k, v in parts.items()},
            "pond_location": {
                "lat": round(float(lat), 7),
                "lon": round(float(lon), 7),
                "elevation_m": round(float(f.filled[r, c]), 2),
                "grid_row_col": [int(r), int(c)],
            },
            "catchment": st,
            "pond_sizing": catch_mod.pond_sizing(
                st["area_m2"], s, rain_mm, st["mean_slope_pct"], rain_src, rain_why),
        }
        if include_geometry and i == 0:
            entry["catchment"]["geometry"] = catch_mod.to_geojson(cm, d)
        results.append(entry)
    tick("catchment")

    return {
        "status": "ok",
        "source": {
            "filename": filename,
            "crs": f"EPSG:{d.epsg}",
            "contours": cs.meta,
        },
        "dem": d.meta,
        "hydrology": f.meta,
        "eligible_cells": int(mask.sum()),
        "recommended_site": results[0],
        "alternative_sites": results[1:],
        "settings_used": s.dict(),
        "timings_sec": timings,
        "total_sec": round(time.time() - t0, 2),
    }

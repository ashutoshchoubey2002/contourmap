"""
Saare tunable numbers yahan. Code ke andar koi magic number nahi.

Phase-2 mein weights ya thresholds badalne ho -- sirf ye file
badalni hai, algorithm ko haath lagane ki zaroorat nahi.
"""

from typing import Optional
from dataclasses import dataclass, field, asdict


@dataclass
class Settings:
    # --- DEM ---
    target_cells: int = 350          # lambi side par itne cells
    cell_min_m: float = 2.0
    cell_max_m: float = 25.0
    max_points: int = 120_000        # isse zyada hue to subsample

    # --- terrain conditioning ---
    smooth_sigma: float = 1.0        # noise ke nakli gaddhe kam karne ko
    fill_epsilon: float = 1e-5       # flat jagah par dhalan banane ko

    # --- level cleaning ---
    level_gap_factor: float = 10.0   # itne guna bada gap = outlier cluster

    # --- pond siting ---
    w_runoff: float = 0.50
    w_flatness: float = 0.25
    w_concavity: float = 0.25
    tpi_radius_cells: int = 15

    border_margin_frac: float = 0.08
    min_catchment_ha: float = 2.0
    max_catchment_frac: float = 0.80
    n_candidates: int = 5
    min_separation_m: float = 400.0

    # --- pond sizing ---
    # None means "work it out": rainfall is looked up for the site (see
    # app/rainfall.py) and the runoff coefficient comes from catchment slope.
    annual_rainfall_mm: Optional[float] = None
    runoff_coefficient: Optional[float] = None
    pond_depth_m: float = 3.0

    def dict(self):
        return asdict(self)


DEFAULTS = Settings()
# ============================================================================
# Area path (map selection) — appended in phase 2.
# Everything above this line is the contour path and is unchanged.
# ============================================================================

import os as _os
import tempfile as _tf

# --- terrain tiles ---
TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TILE_TARGET_CELLS = 600      # long side of the grid; caps cost per request
TILE_ZOOM_MIN = 8
TILE_ZOOM_MAX = 14     
TILE_MIN_CELL_M = 25.0    # new line, right after TILE_ZOOM_MAX      # ~10 m/px; terrarium has no real detail beyond this
TILE_MAX_TILES = 64
TILE_CONCURRENCY = 8
TILE_TIMEOUT_S = 10.0
TILE_RETRIES = 3
TILE_BACKOFF_S = 0.25
TILE_USER_AGENT = "pondapi/2.0"

# --- selection ---
AREA_BUFFER_FRAC = 0.5       # fetch beyond the drawn box so catchments aren't clipped
AREA_MAX_KM2 = 200.0
AREA_MIN_KM2 = 0.05

# --- hydrology ---
SMOOTH_SIGMA = 1.0
FILL_EPSILON = 1e-5
STREAM_MIN_HA = 5.0

# --- siting weights (sum to 1) ---
W_RUNOFF = 0.50
W_FLATNESS = 0.20
W_CONCAVITY = 0.15
W_DEPTH = 0.15
MAX_CANDIDATES = 5
CANDIDATE_MIN_SEP_M = 400.0

# --- pond design ---
POND_DEPTH_M = 3.0
POND_SIDE_SLOPE = 1.5
POND_TOP_AREA_M2 = 4000.0

# --- rainfall ---
RAINFALL_URL = "https://archive-api.open-meteo.com/v1/archive"
RAINFALL_START = "2019-01-01"   # was 2014 — ten years of daily values is a slow response
RAINFALL_END = "2023-12-31"
RAINFALL_GRID_DEG = 0.25
RAINFALL_TTL_S = 30 * 86400
RAINFALL_TIMEOUT_S = 20.0       
RAINFALL_DEFAULT_MM = 1200.0
# Optional offline annual-rainfall GeoTIFF (mm/year, e.g. WorldClim bio12 or an
# IMD mean). Used first when present; skipped silently when absent.
RAINFALL_RASTER = _os.path.join(_os.path.dirname(__file__), "..", "data", "rain_annual_mm.tif")

# --- admission control and caching ---
MAX_CONCURRENT_JOBS = 2      # raise to 4 for a multi-person demo
RETRY_AFTER_S = 5
RESULT_TTL_S = 7 * 86400
RESULT_MEM_MAX = 64
RESULT_SCHEMA_VERSION = "area-1.0"

# Cache location. Defaults to the system temp directory, so this file works
# unchanged on Windows and Linux. On the four-system deployment set
# POND_CACHE to a path every worker can reach, so tiles fetched by one worker
# and results computed by one worker serve requests landing on another.
_cache_base = _os.environ.get("POND_CACHE", _os.path.join(_tf.gettempdir(), "pond"))
TILE_CACHE_DIR = _os.path.join(_cache_base, "tiles")
RESULT_CACHE_DIR = _os.path.join(_cache_base, "results")


# --- exclusion mask (OpenStreetMap) ---
EXCLUSION_ENABLED = True
OSM_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]
OSM_TIMEOUT_S = 12.0      
OSM_TTL_S = 30 * 86400
OSM_CACHE_DIR = _os.path.join(_cache_base, "osm")
MAX_SITE_SLOPE_PCT = 15.0
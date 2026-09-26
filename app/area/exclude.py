"""
Where a pond must not go.

Scoring on runoff and flatness alone will happily recommend digging in the
middle of a village: settlement sits on flat ground at the bottom of a
catchment, which is exactly what the scoring function rewards. Anyone looking
at the result on satellite imagery sees that immediately, and the whole output
loses credibility.

So features that rule out a site are pulled from OpenStreetMap for the drawn
area and rasterised onto the analysis grid as a veto applied before scoring.

Failure here is not fatal. If Overpass is unreachable the analysis still runs
with an empty mask and says so in the response, because a result with a caveat
beats no result during a demo. What it must never do is fail silently — the
response always carries whether the mask was applied.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
from pyproj import CRS, Transformer
from scipy.ndimage import distance_transform_edt

from app import config

# Each class carries its own clearance in metres, because the reasons differ.
# Buildings need room for the bund and the borrow area. Water bodies are
# excluded outright — proposing a pond where a tank already exists is the most
# visible way to look wrong. Roads need enough clearance that the embankment
# does not undercut them.
FEATURE_CLASSES = {
    "building": 50.0,
    "residential": 60.0,
    "water": 30.0,
    "road": 20.0,
    "railway": 30.0,
}

OVERPASS_QUERY = """
[out:json][timeout:{timeout}];
(
  way["building"]({s},{w},{n},{e});
  relation["building"]({s},{w},{n},{e});
  way["landuse"~"residential|commercial|industrial|retail|quarry|landfill"]({s},{w},{n},{e});
  relation["landuse"~"residential|commercial|industrial|retail"]({s},{w},{n},{e});
  way["place"~"village|town|city|hamlet"]({s},{w},{n},{e});
  way["natural"="water"]({s},{w},{n},{e});
  relation["natural"="water"]({s},{w},{n},{e});
  way["landuse"~"reservoir|basin"]({s},{w},{n},{e});
  way["man_made"~"mineshaft|spoil_heap"]({s},{w},{n},{e});
  way["waterway"~"river|canal"]({s},{w},{n},{e});
  way["highway"~"motorway|trunk|primary|secondary|tertiary"]({s},{w},{n},{e});
  way["railway"="rail"]({s},{w},{n},{e});
);
out geom;
"""


class ExclusionResult:
    """The mask plus an account of where it came from.

    Deliberately carries `available`: a caller must be able to tell an empty
    mask meaning "nothing to avoid here" from one meaning "the lookup failed".
    """

    def __init__(self, mask: np.ndarray, counts: dict, available: bool,
                 note: str = "", seconds: float = 0.0, cached: bool = False,
                 elements: int = 0):
        self.elements = elements
        self.mask = mask
        self.counts = counts
        self.available = available
        self.note = note
        self.seconds = seconds
        self.cached = cached

    def summary(self) -> dict:
        return {
            "applied": self.available,
            "elements_returned": self.elements,
            "features": self.counts,
            "excluded_fraction": round(float(self.mask.mean()), 4),
            "seconds": round(self.seconds, 3),
            "cached": self.cached,
            "note": self.note,
        }


# --------------------------------------------------------------- classify

def _classify(tags: dict) -> str | None:
    """Map OSM tags onto one of FEATURE_CLASSES. Order matters: a building
    tagged with a landuse is still a building."""
    if not tags:
        return None
    if tags.get("building"):
        return "building"
    if tags.get("landuse") in ("residential", "commercial", "industrial",
                               "retail", "quarry", "landfill"):
        return "residential"
    if tags.get("man_made") in ("mineshaft", "spoil_heap"):
        return "residential"
    if tags.get("place") in ("village", "town", "city", "hamlet"):
        return "residential"
    if tags.get("natural") == "water" or tags.get("landuse") in ("reservoir", "basin"):
        return "water"
    if tags.get("waterway") in ("river", "canal"):
        return "water"
    if tags.get("railway") == "rail":
        return "railway"
    if tags.get("highway"):
        return "road"
    return None


# --------------------------------------------------------------- fetch

def _cache_path(bounds) -> str:
    # Rounded to ~100 m so a nudged selection reuses the same answer.
    key = hashlib.sha256(
        json.dumps([round(v, 3) for v in bounds]).encode()
    ).hexdigest()[:24]
    return os.path.join(config.OSM_CACHE_DIR, f"{key}.json")


def _fetch_osm(bounds) -> tuple[list, bool]:
    """Return (elements, from_cache). Raises on failure."""
    path = _cache_path(bounds)
    try:
        if time.time() - os.path.getmtime(path) < config.OSM_TTL_S:
            with open(path) as fh:
                return json.load(fh), True
    except (OSError, ValueError):
        pass

    west, south, east, north = bounds
    query = OVERPASS_QUERY.format(
        s=south, w=west, n=north, e=east, timeout=int(config.OSM_TIMEOUT_S)
    )
    body = urllib.parse.urlencode({"data": query}).encode()

    last = None
    # Overpass mirrors rate-limit independently; trying the next one is
    # cheaper than failing, and the query is identical.
    for endpoint in config.OSM_ENDPOINTS:
        try:
            req = urllib.request.Request(
                endpoint, data=body,
                headers={"User-Agent": config.TILE_USER_AGENT},
            )
            with urllib.request.urlopen(req, timeout=config.OSM_TIMEOUT_S) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            elements = data.get("elements", [])
            try:
                os.makedirs(config.OSM_CACHE_DIR, exist_ok=True)
                tmp = f"{path}.{os.getpid()}.tmp"
                with open(tmp, "w") as fh:
                    json.dump(elements, fh)
                os.replace(tmp, path)
            except OSError:
                pass
            return elements, False
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"all Overpass endpoints failed: {last}")


# --------------------------------------------------------------- rasterise

def _grid_lonlat(dem) -> tuple[np.ndarray, np.ndarray]:
    """Longitude and latitude of every cell centre.

    Built in one transform call rather than per cell; on a 300x400 grid the
    per-cell version takes seconds and this takes milliseconds.
    """
    rows, cols = dem.shape
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    x = dem.x0 + cc * dem.cell_size
    y = dem.y0 - rr * dem.cell_size
    tr = Transformer.from_crs(dem.crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = tr.transform(x.ravel(), y.ravel())
    return (np.asarray(lon).reshape(rows, cols),
            np.asarray(lat).reshape(rows, cols))


def _burn_polygon(out: np.ndarray, glon, glat, ring: list) -> None:
    """Even-odd fill of one ring, restricted to its own bounding box.

    The bbox restriction is what makes this usable: a village box can hold a
    thousand buildings, and testing every cell against every building would be
    hundreds of millions of operations. Each building only touches a handful of
    cells.
    """
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    w, e, s, n = min(lons), max(lons), min(lats), max(lats)

    inbox = (glon >= w) & (glon <= e) & (glat >= s) & (glat <= n)
    if not inbox.any():
        return
    rows, cols = np.nonzero(inbox)
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1

    sub_lon = glon[r0:r1, c0:c1]
    sub_lat = glat[r0:r1, c0:c1]
    inside = np.zeros(sub_lon.shape, dtype=bool)
    m = len(ring)
    for i in range(m):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % m]
        if y1 == y2:
            continue
        crosses = (sub_lat > min(y1, y2)) & (sub_lat <= max(y1, y2))
        x_at = x1 + (sub_lat - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (sub_lon < x_at)
    out[r0:r1, c0:c1] |= inside


def _burn_line(out: np.ndarray, dem, points: list) -> None:
    """Mark the cells a way passes through.

    Vertices alone would leave gaps wherever a road runs straight for longer
    than a cell, so each segment is sampled at half-cell spacing. The clearance
    buffer is applied later and does the rest of the work.
    """
    rows, cols = out.shape
    tr = Transformer.from_crs(CRS.from_epsg(4326), dem.crs, always_xy=True)
    xs, ys = tr.transform([p[0] for p in points], [p[1] for p in points])
    xs, ys = np.atleast_1d(xs), np.atleast_1d(ys)

    for i in range(len(xs) - 1):
        dx, dy = xs[i + 1] - xs[i], ys[i + 1] - ys[i]
        steps = max(2, int(np.hypot(dx, dy) / (dem.cell_size * 0.5)) + 1)
        t = np.linspace(0.0, 1.0, steps)
        px, py = xs[i] + dx * t, ys[i] + dy * t
        c = np.round((px - dem.x0) / dem.cell_size).astype(int)
        r = np.round((dem.y0 - py) / dem.cell_size).astype(int)
        ok = (r >= 0) & (c >= 0) & (r < rows) & (c < cols)
        out[r[ok], c[ok]] = True


def _buffer(mask: np.ndarray, metres: float, cell_size: float) -> np.ndarray:
    """Grow a mask outward by a distance.

    A Euclidean distance transform on the inverse gives every cell its distance
    to the nearest marked cell, so thresholding it produces the buffer in one
    pass regardless of buffer width — much cheaper than repeated dilation for
    the wider clearances.
    """
    if not mask.any() or metres <= 0:
        return mask
    dist = distance_transform_edt(~mask, sampling=cell_size)
    return dist <= metres


# --------------------------------------------------------------- entry point

def build(dem, bounds) -> ExclusionResult:
    """Exclusion mask for the analysis grid. True means a pond may not go here."""
    empty = np.zeros(dem.shape, dtype=bool)
    if not config.EXCLUSION_ENABLED:
        return ExclusionResult(empty, {}, False, "exclusion disabled in config")

    started = time.perf_counter()
    try:
        elements, cached = _fetch_osm(bounds)
    except Exception as exc:  # noqa: BLE001
        return ExclusionResult(
            empty, {}, False,
            f"map features unavailable ({type(exc).__name__}); "
            "sites were not screened against settlement or existing water",
            time.perf_counter() - started,
        )

    glon, glat = _grid_lonlat(dem)
    per_class = {k: np.zeros(dem.shape, dtype=bool) for k in FEATURE_CLASSES}
    counts = {k: 0 for k in FEATURE_CLASSES}

    for el in elements:
        cls = _classify(el.get("tags", {}))
        if cls is None:
            continue
        geom = el.get("geometry")
        if not geom:
            continue
        pts = [(g["lon"], g["lat"]) for g in geom if "lon" in g and "lat" in g]
        if len(pts) < 2:
            continue

        closed = len(pts) > 3 and abs(pts[0][0] - pts[-1][0]) < 1e-9 \
                              and abs(pts[0][1] - pts[-1][1]) < 1e-9
        if closed or cls in ("building", "residential", "water"):
            if len(pts) >= 3:
                _burn_polygon(per_class[cls], glon, glat, pts)
            else:
                _burn_line(per_class[cls], dem, pts)
        else:
            _burn_line(per_class[cls], dem, pts)
        counts[cls] += 1

    mask = np.zeros(dem.shape, dtype=bool)
    for cls, raw in per_class.items():
        if raw.any():
            mask |= _buffer(raw, FEATURE_CLASSES[cls], dem.cell_size)

    # A mask covering nearly everything means the selection is inside a town,
    # not that the data is wrong. Better to say so than to return no site.
    note = ""
    fraction = float(mask.mean())
    if fraction > 0.9:
        note = ("almost all of this area is settlement, water or road; "
                "there is little open land to site a pond")

    return ExclusionResult(
        mask, {k: v for k, v in counts.items() if v},
        True, note, time.perf_counter() - started, cached,
        elements=len(elements),
    )


def steep_ground(slope_pct: np.ndarray, limit: float | None = None) -> np.ndarray:
    """Ground too steep to excavate. Cheap, local, and needs no network, so it
    applies even when Overpass is unreachable."""
    limit = config.MAX_SITE_SLOPE_PCT if limit is None else limit
    return np.isfinite(slope_pct) & (slope_pct > limit)
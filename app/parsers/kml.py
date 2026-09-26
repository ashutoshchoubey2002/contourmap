"""
KML / KMZ contour parser.

Design note (extensibility ke liye zaroori):
  Ye module ContourSet return karta hai -- ek plain container.
  Phase-2 mein agar GeoTIFF DEM ya Shapefile support karna ho, to
  bas ek naya parser likhna hai jo wahi ContourSet de. Baaki pura
  pipeline waisa ka waisa chalega.
"""

import io
import math
import zipfile
from dataclasses import dataclass, field
from collections import Counter
from xml.etree import ElementTree as ET

import numpy as np


ELEV_KEYS = {"elev", "elevation", "contour", "level", "height", "z", "value"}


@dataclass
class ContourSet:
    """Parser ka common output. Har parser yahi banata hai."""
    lon: np.ndarray
    lat: np.ndarray
    elev: np.ndarray
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.lon)

    @property
    def bounds(self):
        return (float(self.lon.min()), float(self.lat.min()),
                float(self.lon.max()), float(self.lat.max()))


# ---------------------------------------------------------------

def _strip_ns(tag):
    return tag.split("}")[-1] if "}" in tag else tag


def _find_all(elem, name):
    """Namespace ki parwah kiye bina descendants dhoondo."""
    return [e for e in elem.iter() if _strip_ns(e.tag) == name]


def _open(data, filename=""):
    """bytes ya path -- dono chalega. KMZ ho to unzip kar do."""
    if isinstance(data, (bytes, bytearray)):
        buf = io.BytesIO(data)
        if data[:2] == b"PK":                     # zip signature = KMZ
            with zipfile.ZipFile(buf) as z:
                names = [n for n in z.namelist() if n.lower().endswith(".kml")]
                if not names:
                    raise ValueError("KMZ ke andar koi .kml nahi mila")
                return io.BytesIO(z.read(names[0]))
        buf.seek(0)
        return buf

    if str(filename or data).lower().endswith(".kmz"):
        with zipfile.ZipFile(data) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".kml")]
            return io.BytesIO(z.read(names[0]))
    return open(data, "rb")


# --- elevation resolvers: teen jagah try karo -------------------

def _from_name(pm):
    for n in _find_all(pm, "name"):
        try:
            return float(n.text.strip()), "name"
        except (AttributeError, ValueError):
            continue
    return None, None


def _from_extended(pm):
    for sd in _find_all(pm, "SimpleData") + _find_all(pm, "Data"):
        key = (sd.get("name") or "").strip().lower()
        if key not in ELEV_KEYS:
            continue
        text = sd.text
        vals = _find_all(sd, "value")
        if vals:
            text = vals[0].text
        try:
            return float(text.strip()), f"extended_data:{key}"
        except (AttributeError, ValueError):
            continue
    return None, None


def _from_z(pm):
    for c in _find_all(pm, "coordinates"):
        if not c.text:
            continue
        for tok in c.text.split():
            parts = tok.split(",")
            if len(parts) >= 3:
                try:
                    return float(parts[2]), "z_coordinate"
                except ValueError:
                    pass
    return None, None


RESOLVERS = (_from_name, _from_extended, _from_z)


def _resolve_elev(pm):
    for fn in RESOLVERS:
        v, src = fn(pm)
        if v is not None:
            return v, src
    return None, None


# ---------------------------------------------------------------

def parse(data, filename="", gap_factor=10.0):
    """
    KML/KMZ -> ContourSet, saath mein outlier levels hata ke.
    """
    with _open(data, filename) as fh:
        root = ET.parse(fh).getroot()

    lons, lats, elevs = [], [], []
    src_count = Counter()
    n_pm = n_skip = 0

    for pm in _find_all(root, "Placemark"):
        n_pm += 1
        e, src = _resolve_elev(pm)
        if e is None:
            n_skip += 1
            continue

        got = False
        for c in _find_all(pm, "coordinates"):
            if not c.text:
                continue
            for tok in c.text.split():
                p = tok.split(",")
                if len(p) < 2:
                    continue
                try:
                    lo, la = float(p[0]), float(p[1])
                except ValueError:
                    continue
                lons.append(lo)
                lats.append(la)
                elevs.append(e)
                got = True
        if got:
            src_count[src] += 1
        else:
            n_skip += 1

    if not lons:
        raise ValueError("Is file mein koi contour point nahi mila")

    lon = np.asarray(lons)
    lat = np.asarray(lats)
    elev = np.asarray(elevs)

    lon, lat, elev, clean = _clean_levels(lon, lat, elev, gap_factor)

    meta = {
        "placemarks": n_pm,
        "skipped": n_skip,
        "elevation_sources": dict(src_count),
        "elevation_source_primary": (src_count.most_common(1)[0][0]
                                     if src_count else None),
        "points": int(len(lon)),
        "contour_interval_m": _interval(elev),
        "levels": int(len(np.unique(elev))),
        "elev_min_m": float(elev.min()),
        "elev_max_m": float(elev.max()),
        "relief_m": float(elev.max() - elev.min()),
        "cleaning": clean,
    }
    return ContourSet(lon, lat, elev, meta)


# ---------------------------------------------------------------

def _clean_levels(lon, lat, elev, gap_factor):
    """
    Kabhi-kabhi file mein ek-aadha bekaar level hota hai (jaise 30.0
    jabki baaki sab 267-298). Levels ko clusters mein baanto jahan
    gap normal se bahut bada ho, aur sabse zyada points wala cluster
    rakho. Kisi bhi map par chalega -- koi value hard-coded nahi.
    """
    uniq = np.unique(elev)
    if len(uniq) < 3:
        return lon, lat, elev, {"removed_levels": []}

    gaps = np.diff(uniq)
    typical = float(np.median(gaps))
    thr = max(typical * gap_factor, typical + 1e-9)

    clusters, start = [], 0
    for i, g in enumerate(gaps):
        if g > thr:
            clusters.append(uniq[start:i + 1])
            start = i + 1
    clusters.append(uniq[start:])

    best = max(clusters, key=lambda c: int(np.isin(elev, c).sum()))
    keep = np.isin(elev, best)

    info = {
        "removed_levels": sorted(set(uniq.tolist()) - set(best.tolist())),
        "typical_gap_m": typical,
        "points_removed": int((~keep).sum()),
    }
    return lon[keep], lat[keep], elev[keep], info


def _interval(elev):
    """Unique heights ke differences ka GCD = asli contour interval."""
    uniq = np.unique(elev)
    if len(uniq) < 2:
        return None
    diffs = {int(round(d * 10)) for d in np.diff(uniq)}
    diffs.discard(0)
    if not diffs:
        return None
    g = 0
    for d in diffs:
        g = math.gcd(g, d)
    return g / 10.0

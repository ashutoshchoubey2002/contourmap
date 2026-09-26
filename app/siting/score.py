"""
Pond site scoring.

Har cell ko teen cheezon par 0-1 ka number diya jaata hai, phir
weighted sum. Sabse zyada score wala cell = pond site.

Kuch bhi hard-coded nahi -- na coordinate, na threshold. Sab
Settings se aata hai.
"""

import numpy as np
from scipy.ndimage import uniform_filter


def normalize(a, mask, lo=2, hi=98):
    """
    0-1 mein laao. Percentile clipping isliye ki ek-do outlier cell
    poore scale ko na bigaad de.
    """
    out = np.zeros_like(a, dtype=float)
    v = a[mask]
    if v.size == 0:
        return out
    p_lo, p_hi = np.percentile(v, [lo, hi])
    if p_hi - p_lo < 1e-12:
        return out
    out[mask] = np.clip((v - p_lo) / (p_hi - p_lo), 0, 1)
    return out


def concavity(filled, radius_cells):
    """
    TPI (Topographic Position Index) = cell ki height minus aas-paas
    ki average height.
      negative  = cell aas-paas se neecha  = natural gaddha
    Hum -TPI lete hain, to zyada value = zyada gehra gaddha =
    utni hi khudai mein zyada paani.
    """
    valid = ~np.isnan(filled)
    z0 = np.where(valid, filled, 0.0)
    size = 2 * radius_cells + 1

    num = uniform_filter(z0, size=size, mode="nearest")
    den = uniform_filter(valid.astype(float), size=size, mode="nearest")
    local_mean = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
    return -(filled - local_mean)


def eligibility(dem, flow, settings):
    """Kaunse cells pond ke liye consider kiye ja sakte hain."""
    z = flow.filled
    ny, nx = z.shape
    valid = ~np.isnan(z)

    # kinare ke paas ke cells ka catchment map ke bahar tak jaata hai
    m = int(round(settings.border_margin_frac * min(ny, nx)))
    inner = np.zeros((ny, nx), bool)
    inner[m:ny - m, m:nx - m] = True

    acc_ha = flow.acc * dem.cell_area / 1e4
    big_enough = acc_ha >= settings.min_catchment_ha
    not_whole_map = flow.acc <= settings.max_catchment_frac * valid.sum()

    return valid & inner & big_enough & not_whole_map


def score(dem, flow, settings):
    """Return (score grid, component grids, eligibility mask)."""
    mask = eligibility(dem, flow, settings)

    # log isliye kyunki accumulation 1 se 60000 tak jaata hai --
    # bina log ke sirf trunk stream jeetega, baaki sab 0 ho jayenge
    runoff = normalize(np.log10(np.maximum(flow.acc, 1.0)), mask)
    flatness = 1.0 - normalize(flow.slope, mask)
    conc = normalize(concavity(flow.filled, settings.tpi_radius_cells), mask)

    s = (settings.w_runoff * runoff
         + settings.w_flatness * flatness
         + settings.w_concavity * conc)
    s[~mask] = -np.inf

    return s, {"runoff": runoff, "flatness": flatness, "concavity": conc}, mask


def pick_sites(s, dem, settings):
    """Top-N, lekin ek doosre se kam se kam min_separation_m door."""
    nx = s.shape[1]
    sep = settings.min_separation_m / dem.cell
    flat = s.ravel()

    chosen = []
    for idx in np.argsort(flat)[::-1]:
        if len(chosen) >= settings.n_candidates:
            break
        if not np.isfinite(flat[idx]):
            break
        r, c = divmod(int(idx), nx)
        if all((r - rr) ** 2 + (c - cc) ** 2 >= sep ** 2
               for rr, cc in chosen):
            chosen.append((r, c))
    return chosen

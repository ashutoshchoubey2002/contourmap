"""
DEM -> flow network.

Teen classic hydrology steps:
  1. fill  -- gaddhe bharo, warna paani wahin ruk jaata hai
  2. D8    -- har cell ka paani kis padosi mein jaayega
  3. acc   -- har cell mein upar se kitne cells ka paani aata hai

Sab khud likha hai (library nahi) taaki har line explain ki ja sake.
"""

from dataclasses import dataclass
import heapq

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter


# 8 padosi: E, NE, N, NW, W, SW, S, SE
DR = np.array([0, -1, -1, -1, 0, 1, 1, 1])
DC = np.array([1, 1, 0, -1, -1, -1, 0, 1])
DIST = np.array([1, 2 ** .5, 1, 2 ** .5, 1, 2 ** .5, 1, 2 ** .5])
N8 = np.ones((3, 3), bool)


@dataclass
class FlowNet:
    filled: np.ndarray
    fdir: np.ndarray        # 0-7 = direction, -1 = pit / nodata
    acc: np.ndarray         # cells
    slope: np.ndarray       # m/m
    meta: dict


# ---------------------------------------------------------------

def smooth(z, sigma):
    """
    NaN-safe Gaussian blur.
    Chaal: NaN ko 0 karke blur karo, mask ko bhi blur karo, phir
    divide. Isse NaN wala border andar nahi failta.
    """
    if sigma <= 0:
        return z.copy()
    valid = ~np.isnan(z)
    num = gaussian_filter(np.where(valid, z, 0.0), sigma, mode="nearest")
    den = gaussian_filter(valid.astype(float), sigma, mode="nearest")
    out = np.full_like(z, np.nan)
    ok = den > 1e-6
    out[ok] = num[ok] / den[ok]
    out[~valid] = np.nan
    return out


def fill_depressions(z, eps):
    """
    Priority-flood + epsilon (Barnes et al. 2014).

    Kinare se paani bharna shuru karo. Hamesha sabse neeche wala cell
    pop karo, uske padosi ko kam se kam utna ooncha kar do.

    eps har step par 0.00001 m badha deta hai. Isse bilkul samtal
    jagah par bhi halki dhalan bani rehti hai -- warna D8 ko pata
    nahi chalta paani kidhar jaaye. Itni chhoti value hai ki
    heights par asar nahi padta.
    """
    ny, nx = z.shape
    nodata = np.isnan(z)

    border = np.zeros((ny, nx), bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    near_nodata = binary_dilation(nodata, structure=N8) & ~nodata
    seeds = (~nodata) & (near_nodata | border)

    filled = np.full((ny, nx), np.inf)
    closed = nodata.copy()          # nodata ki alag copy -- warna
                                    # seed cascade ho jaata hai
    heap, tie = [], 0
    for r, c in zip(*np.nonzero(seeds)):
        r, c = int(r), int(c)
        filled[r, c] = z[r, c]
        closed[r, c] = True
        heapq.heappush(heap, (float(z[r, c]), tie, r, c))
        tie += 1

    n_seeds = len(heap)
    while heap:
        e, _, r, c = heapq.heappop(heap)
        for k in range(8):
            rr, cc = r + int(DR[k]), c + int(DC[k])
            if rr < 0 or cc < 0 or rr >= ny or cc >= nx or closed[rr, cc]:
                continue
            new_e = max(float(z[rr, cc]), e + eps)
            filled[rr, cc] = new_e
            closed[rr, cc] = True
            heapq.heappush(heap, (new_e, tie, rr, cc))
            tie += 1

    filled[nodata] = np.nan
    return filled, n_seeds


def d8(filled, cell):
    """8 padosiyon mein sabse tez dhalan wala chuno."""
    ny, nx = filled.shape
    best = np.zeros((ny, nx))
    fdir = np.full((ny, nx), -1, dtype=np.int8)

    BIG = 1e12
    z = np.where(np.isnan(filled), BIG, filled)

    for k in range(8):
        dr, dc = int(DR[k]), int(DC[k])
        shifted = np.full((ny, nx), BIG)
        r0, r1 = max(0, -dr), ny - max(0, dr)
        c0, c1 = max(0, -dc), nx - max(0, dc)
        shifted[r0:r1, c0:c1] = z[r0 + dr:r1 + dr, c0 + dc:c1 + dc]

        with np.errstate(invalid="ignore"):
            slope = (z - shifted) / (DIST[k] * cell)

        better = (slope > best) & np.isfinite(slope) & (shifted < BIG / 2)
        best[better] = slope[better]
        fdir[better] = k

    fdir[np.isnan(filled)] = -1
    return fdir, best


def accumulate(filled, fdir):
    """
    Sabse ooncha cell pehle process karo, apna total neeche wale
    cell mein daal do. Ooncha-pehle isliye ki upar ka paani hamesha
    pehle aata hai -- ek hi pass mein kaam ho jaata hai.
    """
    ny, nx = filled.shape
    acc = np.where(np.isnan(filled), 0.0, 1.0)

    z = filled.ravel()
    valid = np.flatnonzero(~np.isnan(z))
    order = valid[np.argsort(-z[valid])]

    accr, fdr = acc.ravel(), fdir.ravel()
    dr, dc = DR.astype(int), DC.astype(int)

    for idx in order:
        k = fdr[idx]
        if k < 0:
            continue
        r, c = divmod(int(idx), nx)
        rr, cc = r + dr[k], c + dc[k]
        if 0 <= rr < ny and 0 <= cc < nx:
            accr[rr * nx + cc] += accr[idx]
    return acc


# ---------------------------------------------------------------

def analyze(dem, settings):
    z = smooth(dem.z, settings.smooth_sigma)
    filled, n_seeds = fill_depressions(z, settings.fill_epsilon)
    fdir, slope = d8(filled, dem.cell)
    acc = accumulate(filled, fdir)

    valid = ~np.isnan(filled)
    meta = {
        "smooth_sigma": settings.smooth_sigma,
        "outlet_seeds": n_seeds,
        "cells_raised": int(np.nansum((filled - z) > 1e-3)),
        "max_fill_m": round(float(np.nanmax(filled - z)), 2),
        "pits_remaining": int(((fdir < 0) & valid).sum()),
        "mean_slope_pct": round(float(np.nanmean(slope[valid]) * 100), 2),
        "max_accumulation_cells": int(acc.max()),
        "max_accumulation_ha": round(acc.max() * dem.cell_area / 1e4, 1),
    }
    return FlowNet(filled, fdir, acc, slope, meta)

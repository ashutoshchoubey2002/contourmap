"""
Hydrological routing on a metric grid.

This mirrors the method already used on the contour path — priority-flood
depression filling with an epsilon increment, D8 routing, then accumulation —
but operates on an AreaDEM and has no dependency on the contour parsing stage.
Keeping it separate means the working contour pipeline is not touched while the
area path is brought up. Once both are proven, one of the two implementations
can be retired.

Everything is done on flat indices. A 300x400 grid is 120k cells, which is fine
in pure Python provided per-cell work stays small and the inner loops avoid
attribute lookups.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from app import config

# D8 neighbour offsets, clockwise from east. Index into these is the direction
# code stored per cell; -1 means no downslope neighbour (an outlet).
DR = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int8)
DC = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int8)


@dataclass
class FlowResult:
    filled: np.ndarray        # depression-free surface, metres
    direction: np.ndarray     # int8 D8 code per cell, -1 at outlets
    downstream: np.ndarray    # int32 flat index of receiver, -1 at outlets
    accumulation: np.ndarray  # float32 contributing cell count (inclusive)
    order: np.ndarray         # int32 flat indices, highest filled elevation first
    fill_depth: np.ndarray    # filled - original, metres
    cells_raised: int
    max_fill: float


def smooth(dem) -> np.ndarray:
    """Gaussian smoothing that neither reads from nor writes into nodata.

    Tile elevation is quantised to ~0.004 m steps and carries resampling noise.
    Left alone, that noise creates thousands of one-cell pits and the drainage
    network fragments.
    """
    z = dem.elevation.astype(np.float64)
    valid = np.isfinite(z)
    filled = np.where(valid, z, 0.0)
    num = gaussian_filter(filled, config.SMOOTH_SIGMA, mode="nearest")
    den = gaussian_filter(valid.astype(np.float64), config.SMOOTH_SIGMA, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(valid, out, np.nan)


def fill_depressions(z: np.ndarray, eps: float | None = None) -> tuple[np.ndarray, int, float]:
    """Priority-flood (Barnes, Lehman & Mulla 2014) with epsilon increment.

    Floods inward from every cell where water can leave the grid — the border,
    and any cell touching nodata. Because cells come off the queue in ascending
    elevation, each is assigned the lowest level from which water reaching it
    can still escape, which is the filled surface by definition.

    The epsilon matters more than it looks. Without it, filled regions are
    exactly flat, a flat cell has no steepest descent, and D8 has nothing to
    choose. On gentle terrain that fragments the network badly.
    """
    eps = config.FILL_EPSILON if eps is None else eps
    rows, cols = z.shape
    filled = np.full(z.shape, np.inf, dtype=np.float64)
    closed = ~np.isfinite(z)

    heap: list[tuple[float, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if closed[r, c]:
                continue
            edge = r == 0 or c == 0 or r == rows - 1 or c == cols - 1
            if not edge:
                # adjacent to nodata counts as an outlet
                r0, r1 = max(r - 1, 0), min(r + 2, rows)
                c0, c1 = max(c - 1, 0), min(c + 2, cols)
                edge = bool(closed[r0:r1, c0:c1].any())
            if edge:
                filled[r, c] = z[r, c]
                closed[r, c] = True
                heapq.heappush(heap, (float(z[r, c]), r, c))

    while heap:
        e, r, c = heapq.heappop(heap)
        for k in range(8):
            nr, nc = r + int(DR[k]), c + int(DC[k])
            if nr < 0 or nc < 0 or nr >= rows or nc >= cols or closed[nr, nc]:
                continue
            ne = max(float(z[nr, nc]), e + eps)
            filled[nr, nc] = ne
            closed[nr, nc] = True
            heapq.heappush(heap, (ne, nr, nc))

    filled = np.where(np.isfinite(z), filled, np.nan)
    depth = np.where(np.isfinite(z), filled - z, 0.0)
    raised = int((depth > 1e-4).sum())
    return filled, raised, float(np.nanmax(depth) if raised else 0.0)


def d8(filled: np.ndarray, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Steepest-descent direction, distance-weighted.

    Diagonal neighbours are sqrt(2) further away, so comparing raw drops biases
    flow into the diagonals. Slope is compared instead.
    """
    rows, cols = filled.shape
    dist = np.where((DR != 0) & (DC != 0), np.sqrt(2.0), 1.0) * cell_size

    best_slope = np.zeros(filled.shape, dtype=np.float64)
    direction = np.full(filled.shape, -1, dtype=np.int8)

    for k in range(8):
        # int() matters: DR/DC are int8 and numpy will not widen them against
        # a Python int, so `cols + min(0, -DC[k])` overflows on any grid
        # wider than 127 columns.
        dr, dc = int(DR[k]), int(DC[k])
        shifted = np.full(filled.shape, np.nan)
        r0, r1 = max(0, dr), rows + min(0, dr)
        c0, c1 = max(0, dc), cols + min(0, dc)
        sr0, sr1 = max(0, -dr), rows + min(0, -dr)
        sc0, sc1 = max(0, -dc), cols + min(0, -dc)
        shifted[sr0:sr1, sc0:sc1] = filled[r0:r1, c0:c1]

        with np.errstate(invalid="ignore"):
            slope = (filled - shifted) / dist[k]
            better = np.isfinite(slope) & (slope > best_slope)
        best_slope = np.where(better, slope, best_slope)
        direction = np.where(better, k, direction).astype(np.int8)

    direction = np.where(np.isfinite(filled), direction, -1).astype(np.int8)

    idx = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    rr, cc = np.divmod(idx, cols)
    downstream = np.full(rows * cols, -1, dtype=np.int32)
    has = direction >= 0
    k = np.clip(direction, 0, 7)
    nr = rr + DR[k]
    nc = cc + DC[k]
    ok = has & (nr >= 0) & (nc >= 0) & (nr < rows) & (nc < cols)
    downstream[idx[ok]] = (nr[ok] * cols + nc[ok]).astype(np.int32)
    return direction, downstream


def accumulate(filled: np.ndarray, downstream: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """Contributing-cell count, by processing cells highest first.

    Sorting by elevation is a valid topological order because every receiver is
    strictly lower than its donor after filling with epsilon. That guarantee is
    what makes a single pass sufficient.
    """
    flat = filled.ravel()
    valid = np.isfinite(flat)
    order = np.argsort(np.where(valid, -flat, np.inf), kind="stable").astype(np.int32)
    order = order[: int(valid.sum())]

    acc = np.where(valid, 1.0, 0.0).astype(np.float64)
    down = downstream
    for i in order:
        d = down[i]
        if d >= 0:
            acc[d] += acc[i]
    return acc.reshape(filled.shape).astype(np.float32), order


def route(dem) -> FlowResult:
    z = smooth(dem)
    filled, raised, maxfill = fill_depressions(z)
    direction, downstream = d8(filled, dem.cell_size)
    acc, order = accumulate(filled, downstream)
    return FlowResult(
        filled=filled,
        direction=direction,
        downstream=downstream,
        accumulation=acc,
        order=order,
        fill_depth=np.where(np.isfinite(z), filled - z, 0.0),
        cells_raised=raised,
        max_fill=maxfill,
    )


# --------------------------------------------------------------- catchment

def _children_index(downstream: np.ndarray, n: int):
    """CSR-style reverse adjacency: for each cell, the cells that drain into it.

    Built once with an argsort rather than by appending to per-cell lists,
    which would allocate 120k Python lists and dominate the runtime.
    """
    src = np.arange(n, dtype=np.int32)
    has = downstream >= 0
    tgt = downstream[has]
    src = src[has]
    order = np.argsort(tgt, kind="stable")
    tgt_sorted = tgt[order]
    src_sorted = src[order]
    starts = np.searchsorted(tgt_sorted, np.arange(n + 1))
    return starts, src_sorted


def delineate(downstream: np.ndarray, shape: tuple[int, int], pour: int
              ) -> np.ndarray:
    """Every cell whose flow path reaches pour. Iterative upstream walk."""
    n = shape[0] * shape[1]
    starts, children = _children_index(downstream, n)
    mask = np.zeros(n, dtype=bool)
    stack = [int(pour)]
    mask[pour] = True
    while stack:
        c = stack.pop()
        for j in range(starts[c], starts[c + 1]):
            u = int(children[j])
            if not mask[u]:
                mask[u] = True
                stack.append(u)
    return mask.reshape(shape)


def stream_mask(accumulation: np.ndarray, cell_area: float,
                min_ha: float | None = None) -> np.ndarray:
    """Cells carrying more than a threshold contributing area — the channels."""
    min_ha = config.STREAM_MIN_HA if min_ha is None else min_ha
    min_cells = (min_ha * 10_000.0) / cell_area
    return np.isfinite(accumulation) & (accumulation >= min_cells)


def slope_percent(filled: np.ndarray, cell_size: float) -> np.ndarray:
    """Horn slope in percent."""
    gy, gx = np.gradient(np.nan_to_num(filled, nan=np.nanmean(filled)), cell_size)
    return np.hypot(gx, gy) * 100.0
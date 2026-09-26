"""
HTTP surface for the area path.

A router rather than edits to main.py, so wiring it up is two lines and the
existing /analyzeContour route is untouched:

    from app.area_routes import router as area_router
    app.include_router(area_router)

Optionally also serve the page from the API, which avoids CORS entirely:

    from fastapi.staticfiles import StaticFiles
    app.mount("/app", StaticFiles(directory="frontend", html=True), name="ui")
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from app import config, limits
from app.area_pipeline import analyze_area
from app.dem import tiles

router = APIRouter(tags=["area"])


class AreaRequest(BaseModel):
    # Accepts either a bare ring of [lon, lat] pairs or a GeoJSON geometry,
    # because Leaflet.draw produces the latter and hand-written clients the
    # former.
    polygon: list[list[float]] | None = Field(
        default=None, description="Ring of [lon, lat] pairs"
    )
    geometry: dict | None = Field(default=None, description="GeoJSON Polygon")
    rainfall_mm: float | None = Field(
        default=None, gt=0, lt=10000,
        description="Annual rainfall. Omitted means look it up from ERA5.",
    )
    runoff_coefficient: float | None = Field(
        default=None, gt=0, le=1,
        description="Rational-method C. Omitted means derive from slope.",
    )
    refresh: bool = Field(default=False, description="Bypass the result cache")

    def ring(self) -> list[tuple[float, float]]:
        raw = self.polygon
        if raw is None and self.geometry:
            if self.geometry.get("type") != "Polygon":
                raise ValueError("geometry must be a Polygon")
            raw = self.geometry["coordinates"][0]
        if not raw:
            raise ValueError("provide polygon or geometry")
        ring = [(float(p[0]), float(p[1])) for p in raw]
        # Leaflet closes the ring; a duplicated last vertex breaks nothing but
        # adds a zero-length edge to every point-in-polygon test.
        if len(ring) > 1 and ring[0] == ring[-1]:
            ring = ring[:-1]
        if len(ring) < 3:
            raise ValueError("polygon needs at least three distinct vertices")
        for lon, lat in ring:
            if not (-180 <= lon <= 180 and -85 <= lat <= 85):
                raise ValueError(f"coordinate out of range: {lon}, {lat}")
        return ring


@router.post("/analyzeArea", summary="Site a pond within a drawn area")
def analyze(req: AreaRequest, response: Response):
    try:
        ring = req.ring()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        result = analyze_area(
            ring,
            rainfall_mm=req.rainfall_mm,
            runoff_coefficient=req.runoff_coefficient,
            use_cache=not req.refresh,
        )
    except limits.Busy as exc:
        # 429 rather than a slow 200: the balancer can retry a refusal
        # elsewhere, but it cannot do anything useful with a request that is
        # merely taking too long.
        # Headers must ride on the exception: anything set on `response` is
        # discarded once HTTPException is raised.
        raise HTTPException(
            status_code=429,
            detail="worker at capacity, retry shortly",
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except tiles.TileError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"elevation data unavailable: {exc.reason}",
        ) from exc

    response.headers["X-Cache"] = "hit" if result.get("cached") else "miss"
    return result


@router.get("/limits", summary="Configured operating limits")
def limits_info():
    return {
        "max_area_km2": config.AREA_MAX_KM2,
        "min_area_km2": config.AREA_MIN_KM2,
        "max_concurrent_jobs": config.MAX_CONCURRENT_JOBS,
        "target_grid_cells": config.TILE_TARGET_CELLS,
        "zoom_range": [config.TILE_ZOOM_MIN, config.TILE_ZOOM_MAX],
        "buffer_fraction": config.AREA_BUFFER_FRAC,
        "tiles": tiles.cache_stats(),
    }
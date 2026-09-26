"""
FastAPI app for village pond planning.

Two ways in, one result shape:

    POST /analyzeContour   -- upload a KML/KMZ, receive catchment JSON
    POST /findCatchment    -- same route, alternative name
    POST /analyzeArea      -- draw a polygon on a map, receive the same JSON

    GET  /health           -- status and worker load, read by the load balancer
    GET  /limits           -- configured operating limits
    GET  /defaults         -- contour-path defaults
    GET  /app/             -- the browser interface
    GET  /docs             -- auto-generated API documentation
"""

import dataclasses
from typing import Optional
from xml.etree import ElementTree as ET

from fastapi import (FastAPI, File, UploadFile, HTTPException, Query,
                     Request)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import limits
from .area_routes import router as area_router
from .config import Settings, DEFAULTS
from .pipeline import analyze


app = FastAPI(
    title="Village Pond Planning API",
    version="0.2.0",
    description=(
        "Site a rainwater harvesting pond and delineate the catchment that "
        "drains to it. Terrain comes either from an uploaded contour map or "
        "from elevation tiles for an area drawn on a map. Nothing is "
        "hard-coded; every value is derived from the terrain."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# The area path lives in its own router so the contour path above is
# untouched. /health is deliberately NOT defined there -- it is defined once,
# below, so there is no ambiguity about which one FastAPI registers.
app.include_router(area_router)

MAX_MB = 50
ALLOWED = (".kml", ".kmz")


@app.get("/")
def root():
    return {
        "name": "Village Pond Planning API",
        "docs": "/docs",
        "interface": "/app/",
        "endpoints": [
            "/analyzeContour", "/findCatchment", "/analyzeArea",
            "/health", "/limits", "/defaults",
        ],
    }


@app.get("/health")
def health():
    """Status plus this worker's own load.

    The load figures matter on the multi-system deployment: all the systems
    are containers on one host, so machine CPU is identical everywhere and
    tells the balancer nothing about which worker is actually busy. `load` is
    the fraction of this process's analysis slots in use.
    """
    return {"status": "ok", "version": app.version, **limits.load_report()}


@app.get("/defaults")
def defaults():
    return DEFAULTS.dict()


@app.post("/analyzeContour")
async def analyze_contour(
    request: Request,
    file: UploadFile = File(None, description="Contour map (.kml or .kmz)"),
    n_candidates: int = Query(5, ge=1, le=20),
    min_catchment_ha: float = Query(2.0, gt=0),
    target_cells: int = Query(350, ge=100, le=800),
    smooth_sigma: float = Query(1.0, ge=0, le=5),
    rainfall_mm: Optional[float] = Query(
        None, gt=0, description="Annual rainfall (mm). Omit to look it up for the site."),
    runoff_coefficient: Optional[float] = Query(
        None, gt=0, le=1, description="Runoff coefficient. Omit to derive it from slope."),
    pond_depth_m: float = Query(3.0, gt=0),
    include_geometry: bool = Query(True),
):
    # The client may name the upload field anything -- file, contour_map,
    # kml, upload. Take whichever form field carries a filename so that
    # the endpoint works with any client.
    if file is None:
        form = await request.form()
        for value in form.values():
            if hasattr(value, "filename") and value.filename:
                file = value
                break

    if file is None:
        raise HTTPException(
            400,
            "No file received. Send a .kml or .kmz as multipart/form-data."
        )

    name = (file.filename or "").lower()
    if not name.endswith(ALLOWED):
        raise HTTPException(400, f"Only {ALLOWED} files are accepted")

    data = await file.read()
    if not data:
        raise HTTPException(400, "File is empty")
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds the {MAX_MB} MB limit")

    settings = dataclasses.replace(
        DEFAULTS,
        n_candidates=n_candidates,
        min_catchment_ha=min_catchment_ha,
        target_cells=target_cells,
        smooth_sigma=smooth_sigma,
        annual_rainfall_mm=rainfall_mm,
        runoff_coefficient=runoff_coefficient,
        pond_depth_m=pond_depth_m,
    )

    try:
        return analyze(data, file.filename, settings, include_geometry)
    except ET.ParseError as e:
        raise HTTPException(400, f"Could not parse the file: {e}")
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")


app.add_api_route("/findCatchment", analyze_contour,
                  methods=["POST"], name="find_catchment")

# Mounted last. A mount swallows every path beneath it, so mounting before the
# API routes are registered would shadow them.
app.mount("/app", StaticFiles(directory="frontend", html=True), name="ui")
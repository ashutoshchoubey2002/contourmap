# Village Pond Planning API — Contour Analysis Phase

Contour map (KML/KMZ) leta hai, terrain analyze karta hai, aur pond ke liye
suitable jagah + uska catchment area batata hai.

Koi coordinate, threshold, ya result hard-coded nahi hai. Sab kuch input
file se derive hota hai.

---

## Install

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` — Swagger UI apne aap ban jaata hai.

API ke bina test karne ke liye:

```bash
python run_local.py contours_1m.kml     # result.json banata hai
```

---

## API

### `POST /analyzeContour`

Alias: `POST /findCatchment` (dono ek hi hain)

**Request** — `multipart/form-data`

| field | type | required |
|---|---|---|
| `file` | `.kml` or `.kmz`, max 50 MB | yes |

**Query parameters** (sab optional)

| param | default | matlab |
|---|---|---|
| `n_candidates` | 5 | kitne sites return karne hain |
| `min_catchment_ha` | 2.0 | isse chhota catchment reject |
| `target_cells` | 350 | grid ki resolution; zyada = barik par slow |
| `smooth_sigma` | 1.0 | DEM smoothing; noisy data ke liye badhao |
| `rainfall_mm` | 1200 | annual rainfall (phase-2 mein API se) |
| `runoff_coefficient` | 0.25 | barish ka kitna hissa behta hai |
| `pond_depth_m` | 3.0 | proposed depth |
| `include_geometry` | true | catchment polygon chahiye ya nahi |

```bash
curl -X POST "http://127.0.0.1:8000/analyzeContour?n_candidates=3" \
     -F "file=@contours_1m.kml"
```

**Response 200**

```jsonc
{
  "status": "ok",
  "source": {
    "filename": "contours_1m.kml",
    "crs": "EPSG:32644",
    "contours": {
      "placemarks": 2712,
      "elevation_source_primary": "name",
      "points": 160468,
      "contour_interval_m": 1.0,
      "relief_m": 31.0,
      "cleaning": { "removed_levels": [30.0] }
    }
  },
  "dem": { "cell_size_m": 8.1, "grid_rows": 325, "coverage_pct": 97.1 },
  "hydrology": {
    "cells_raised": 34917,
    "pits_remaining": 56,
    "max_accumulation_ha": 394.1
  },
  "recommended_site": {
    "rank": 1,
    "score": 0.942,
    "score_components": { "runoff": 1.0, "flatness": 0.93, "concavity": 0.88 },
    "pond_location": { "lat": 21.24293, "lon": 81.28699, "elevation_m": 268.4 },
    "catchment": {
      "area_ha": 378.3,
      "mean_slope_pct": 3.9,
      "relief_m": 29.5,
      "truncated_by_map_edge": true,
      "geometry": { "type": "Polygon", "coordinates": [[[81.28, 21.24], "..."]] }
    },
    "pond_sizing": { "estimated_annual_runoff_m3": 1134900.0 }
  },
  "alternative_sites": ["..."]
}
```

`geometry` GeoJSON hai — frontend Leaflet par seedha `L.geoJSON()` se daal sakta hai.

**Errors**

| code | kab |
|---|---|
| 400 | galat extension, khaali file, kharab XML |
| 413 | 50 MB se badi file |
| 422 | file to padhi gayi par koi suitable site nahi mila |
| 500 | kuch aur |

---

## Catchment estimation — approach

Paani hamesha neeche behta hai. Agar har jagah ki height pata ho, to
poora drainage network nikala ja sakta hai. Six stages:

**1. Parse.** KML se contour vertices nikalte hain. Elevation teen jagah
ho sakti hai — Placemark ka `<name>`, `ExtendedData`, ya coordinate ka
Z component. Teenon try karte hain aur jo mile use record karte hain.
Sample file mein `<name>` se aati hai.

Ek generic cleaning bhi hai: kabhi file mein ek-aadha bekaar level hota
hai. Levels ke gaps ka median nikaal ke, jahan gap 10× se bada ho wahan
cluster todte hain, aur sabse zyada points wala cluster rakhte hain.
Sample file mein isne `30.0` hata diya (baaki sab 267–298 the).

**2. DEM.** Vertices ko UTM mein reproject karte hain (zone data ke apne
centroid se), taaki area meters mein sahi mile. Phir Delaunay
triangulation par linear interpolation se regular grid banate hain.
Cell size extent se derive hota hai — `max(width,height)/350`, 2–25 m
mein clamp.

**3. Conditioning.** Contour-interpolated DEM mein bahut noise hota hai
jo hazaaron nakli micro-pits banata hai. Halka Gaussian smoothing
(sigma=1) lagate hain, phir priority-flood se depressions bharte hain.

Filling mein epsilon (1e-5 m) add karte hain — isse bilkul flat jagah
par bhi halki dhalan bani rehti hai. Ye zaroori tha kyunki sample
terrain flat hai (31 m relief in 3.2 km, ~1% average slope) aur bina
epsilon ke D8 flats par direction decide nahi kar paata.

**4. Flow.** D8: har cell ke 8 padosiyon mein sabse tez dhalan wala
chunte hain (diagonal ki doori √2 se normalize). Phir accumulation:
cells ko descending elevation mein sort karke, har cell apna total
downstream cell mein add kar deta hai. Ek pass, O(n log n).

**5. Site selection.** Har eligible cell ko teen normalized (0–1)
metrics par score karte hain:

| metric | weight | kyun |
|---|---|---|
| `log10(accumulation)` | 0.50 | kitna paani aata hai. Log isliye ki range 1–60000 hai |
| `1 − slope` | 0.25 | samtal zameen = kam khudai, zyada storage |
| `−TPI` | 0.25 | natural gaddha = mufat storage |

TPI = cell ki height minus 15-cell neighbourhood ki average. Negative
matlab depression.

Normalization 2nd–98th percentile clipping ke saath, taaki outlier cell
scale na bigaade. Eligibility: nodata nahi, border ke 8% mein nahi
(unka catchment truncate hota hai), catchment ≥ 2 ha, aur ≤ 80% of map.

Top-N greedy select hote hain, ek doosre se ≥ 400 m door.

**6. Delineation.** Pour point se upstream BFS. Padosi mujhme behta hai
agar uski flow direction ulti ho — `fdir[neighbour] == (k+4)%8`. Visited
set hi catchment hai. Area = cells × cell_area. Boundary matplotlib
contour se nikaal ke Douglas-Peucker se simplify karte hain, phir
lon/lat mein wapas.

---

## Known limitations

- **Catchment truncation.** Sample map par recommended site ka catchment
  DEM ke kinare tak pahunchta hai, matlab asli catchment map se bahar
  jaata hai. Response mein `truncated_by_map_edge: true` aata hai, aur
  us case mein area **lower bound** hai. Ye contour map ki limitation
  hai, algorithm ki nahi.

- **Runoff weight ka jhukav.** `w_runoff = 0.50` hone se scoring sabse
  bade nale ko chunti hai. Sample par 378 ha catchment mila, jo ek
  village pond ke liye bada hai (typically 10–100 ha). Weights
  `app/config.py` mein hain aur badle ja sakte hain.

- **Rainfall assumed hai.** `pond_sizing` abhi fixed rainfall aur
  runoff coefficient use karta hai. Phase-2 mein Open-Meteo / IMD API
  se aayega — interface waisa hi rahega.

- **Performance.** Priority-flood pure Python hai. 8.5 km² @ 8 m par
  ~1–2 min. Bade maps ke liye `target_cells` kam karo, ya
  `richdem`/`pysheds` par shift karo.

---

## Structure

```
app/
  config.py              saare tunable numbers
  parsers/kml.py         KML/KMZ -> ContourSet
  terrain/dem.py         ContourSet -> DEM
  terrain/flow.py        fill, D8, accumulation
  siting/score.py        cell scoring + site selection
  siting/catchment.py    delineation, GeoJSON, sizing
  pipeline.py            stages ko jodta hai
  main.py                FastAPI
```

**Extensibility.** Har stage ek defined object leta aur deta hai
(`ContourSet` → `DEM` → `FlowNet`). Phase-2 mein GeoTIFF DEM support
karna ho, to bas `parsers/geotiff.py` chahiye jo wahi `ContourSet`/`DEM`
return kare — downstream kuch nahi badlega. Rainfall API `pond_sizing()`
mein aayega. Weights `config.py` mein hain, code mein nahi.

---

## AI tools disclosure

Is project mein Claude (Anthropic) ka use kiya gaya hai — algorithm
design discussion, code drafting, aur debugging ke liye. Do bugs isi
process mein pakde gaye: priority-flood ke seeding mein `closed` array
ka galat reuse, aur NumPy 2.0 mein 2D `np.cross` ka hataya jaana. Saare
algorithms (priority-flood, D8, accumulation, upstream BFS,
Douglas-Peucker) khud implement kiye gaye hain, kisi hydrology library
se nahi liye — taaki har line explain ki ja sake.

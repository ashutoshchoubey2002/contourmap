"""
API ke bina, seedha file par pipeline chalane ke liye.
Debugging aur report ke screenshots banane mein kaam aata hai.

    py run_local.py contours_1m.kml
"""
import sys, json
from app.pipeline import analyze

if len(sys.argv) < 2:
    print("Usage: py run_local.py <file.kml>")
    sys.exit(1)

path = sys.argv[1]
with open(path, "rb") as f:
    out = analyze(f.read(), path)

with open("result.json", "w") as f:
    json.dump(out, f, indent=2)

r = out["recommended_site"]
print(json.dumps({k: v for k, v in out.items()
                  if k in ("dem", "hydrology", "total_sec")}, indent=2))
print("\nRecommended:")
print(f"  lat/lon   : {r['pond_location']['lat']}, {r['pond_location']['lon']}")
print(f"  catchment : {r['catchment']['area_ha']} ha")
print(f"  truncated : {r['catchment']['truncated_by_map_edge']}")
print("\nsaved: result.json")

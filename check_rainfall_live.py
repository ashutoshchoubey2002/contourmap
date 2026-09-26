"""Run from your project root:  python check_rainfall_live.py
Hits each rainfall source for real and prints what happened."""
import logging
from app import rainfall

logging.basicConfig(level=logging.WARNING, format="%(message)s")
LAT, LON = 21.19, 81.28   # change to your site

for name, fn, snapped in rainfall.SOURCES:
    rainfall._cache.clear()
    args = rainfall._key(LAT, LON) if snapped else (LAT, LON)
    try:
        print(f"OK    {name:6s} {fn(*args):8.1f} mm/yr")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL  {name:6s} {rainfall._describe(e)}")

rainfall._cache.clear()
print("\nCombined:", rainfall.annual_rainfall_mm(LAT, LON))
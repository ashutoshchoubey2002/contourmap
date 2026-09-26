import urllib.request, ssl, traceback

url = ("https://archive-api.open-meteo.com/v1/archive?latitude=21.25&longitude=81.25"
       "&start_date=2023-01-01&end_date=2023-12-31&daily=precipitation_sum&timezone=UTC")

for label, req in [
    ("plain", urllib.request.Request(url)),
    ("with UA", urllib.request.Request(url, headers={"User-Agent": "pondapi/2.0"})),
    ("browser UA", urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})),
]:
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            print(label, "-> OK", len(r.read()), "bytes")
    except Exception as e:
        print(label, "-> FAILED:", type(e).__name__, e)

print("proxies seen by python:", urllib.request.getproxies())
print("openssl:", ssl.OPENSSL_VERSION)
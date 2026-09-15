"""Cache only disease series exposed in Semarang's public dashboard selectors."""

import concurrent.futures
import hashlib
import json
from pathlib import Path

import requests

OUT = Path("../outputs/extensive_mining/semarang")


def fetch(pair):
    variable, year = pair
    url = f"https://lekminkes.dinkes.semarangkota.go.id/graph/line/{variable}?tahun={year}"
    item = {"url": url, "variable": variable, "year": year}
    try:
        path = OUT / f"{variable}_{year}.json"
        if path.exists():
            item.update(
                status="available",
                path=str(path.resolve()),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            return item
        r = requests.get(url, timeout=45)
        r.raise_for_status()
        payload = r.json()
        path = OUT / f"{variable}_{year}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False))
        item.update(
            status="available",
            path=str(path.resolve()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    except Exception as e:
        item.update(status="failed", error=str(e))
    return item


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(
            pool.map(
                fetch,
                [
                    (v, y)
                    for v in ["dbd", "kasus_malaria", "dbd_mingguan"]
                    for y in range(2019, 2027)
                ],
            )
        )
    (OUT / "manifest.json").write_text(json.dumps(results, indent=2))
    for x in results:
        print(x["variable"], x["year"], x["status"])

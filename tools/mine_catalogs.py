"""Cache public CKAN disease catalogs with complete pagination and failure receipts."""

import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

import requests

OUT = Path("../outputs/extensive_mining/catalogs")
OUT.mkdir(parents=True, exist_ok=True)
CATALOGS = [
    "https://data.malangkota.go.id",
    "https://data.jatengprov.go.id",
    "https://satudata.asahankab.go.id",
    "https://opendata.kebumenkab.go.id",
    "https://data.semarangkota.go.id",
    "https://data.bandaacehkota.go.id",
    "https://data.bandung.go.id",
    "https://data.bantulkab.go.id",
    "https://data.jogjakota.go.id",
    "https://data.surakarta.go.id",
    "https://data.ntbprov.go.id",
    "https://data.kulonprogokab.go.id",
]
QUERIES = ["DBD", "dengue", "malaria", "demam berdarah"]


def mine(base):
    session = requests.Session()
    session.headers["User-Agent"] = "IndonesiaDiseaseDataResearch/1.0 (public aggregate data)"
    packages, receipts = {}, []
    for query in QUERIES:
        start = 0
        receipt = {"catalog": base, "query": query, "retrieved": 0}
        try:
            while True:
                r = session.get(
                    base + "/api/3/action/package_search",
                    params={"q": query, "rows": 100, "start": start},
                    timeout=(10, 35),
                )
                r.raise_for_status()
                payload = r.json()
                if not payload.get("success"):
                    raise ValueError("CKAN search did not succeed")
                result = payload["result"]
                receipt["reported"] = result["count"]
                batch = result["results"]
                for p in batch:
                    p["_catalog"] = base
                    packages[p["id"]] = p
                start += len(batch)
                receipt["retrieved"] = start
                if not batch or start >= result["count"]:
                    break
                time.sleep(0.25)
            receipt["status"] = "complete"
        except Exception as exc:
            receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        receipts.append(receipt)
        if receipt["status"] == "failed" and not packages:
            break
    target = OUT / (hashlib.sha256(base.encode()).hexdigest()[:12] + ".json")
    target.write_text(
        json.dumps(
            {"catalog": base, "packages": list(packages.values()), "receipts": receipts},
            ensure_ascii=False,
        )
    )
    print(base, len(packages), [r["status"] for r in receipts], flush=True)
    return receipts


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        receipts = [r for group in pool.map(mine, CATALOGS) for r in group]
    (OUT / "receipts.json").write_text(json.dumps(receipts, indent=2))

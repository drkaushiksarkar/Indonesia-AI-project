"""Harvest anonymous public SKDR charts for exposed years and administrative filters.

Checkpoint each response. Six workers, session reuse, no login or hidden API access.
Raw zero-filled chart slots are retained as unverified, never assumed confirmed absence.
"""

import concurrent.futures
import hashlib
import json
import re
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://surkarkes.kemkes.go.id"
ROOT = Path("../outputs/extensive_mining/skdr_nationwide")
LOCAL = threading.local()


def session():
    if not hasattr(LOCAL, "session"):
        s = requests.Session()
        r = s.get(BASE + "/dashboard-skdr-dev", timeout=45)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        LOCAL.token = soup.select_one('meta[name="csrf-token"]')["content"]
        LOCAL.session = s
    return LOCAL.session


def discover_locations():
    """Read only province/district options exposed by the anonymous public form."""
    cache = ROOT / "locations.json"
    if cache.exists():
        return json.loads(cache.read_text())
    s = session()
    page = s.get(BASE + "/dashboard-skdr-dev", timeout=45)
    page.raise_for_status()
    (ROOT / "dashboard.html").write_text(page.text)
    soup = BeautifulSoup(page.text, "html.parser")
    locations = [
        {"code": "", "name": "Indonesia", "level": "admin0", "province": "", "district": ""}
    ]
    for option in soup.select("#dashboardskk-kd_prop option[value]"):
        province = option["value"]
        if not province:
            continue
        locations.append(
            {
                "code": province,
                "name": option.get_text(strip=True),
                "level": "admin1",
                "province": province,
                "district": "",
            }
        )
        response = s.post(
            BASE + "/index.php/kabupaten-list",
            data={"_csrf": LOCAL.token, "prov": province},
            timeout=45,
        )
        response.raise_for_status()
        (ROOT / f"districts_{province}.html").write_text(response.text)
        for district in BeautifulSoup(response.text, "html.parser").select("option[value]"):
            if district["value"]:
                locations.append(
                    {
                        "code": district["value"],
                        "name": district.get_text(strip=True),
                        "level": "admin2",
                        "province": province,
                        "district": district["value"],
                    }
                )
    cache.write_text(json.dumps(locations, indent=2, ensure_ascii=False))
    return locations


def fetch(item):
    disease, year, loc = item
    key = f"{disease}_{year}_{loc['code'] or 'national'}"
    path = ROOT / "charts" / f"{key}.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("status") == "parsed":
            return old
    record = {
        "disease_code": disease,
        "year": year,
        **loc,
        "retrieved_at": datetime.now(UTC).isoformat(),
    }
    try:
        s = session()
        payload = {
            "_csrf": LOCAL.token,
            "kode_penyakit": disease,
            "tahun": "",
            "kd_prop": loc["province"],
            "kd_kab": loc["district"],
            "jenis_waktu": "2",
            "minggu1": "2023-W01",
            "minggu2": "2026-W37",
            "tahun1": "",
            "tahun2": "",
            "multi_penyakit": "0",
        }
        url = BASE + "/index.php/api-skk/grafik-suspek-ibs"
        r = s.post(url, data=payload, timeout=(10, 45))
        r.raise_for_status()
        data = [
            {"label": label, "cases": float(v), "klb": float(k)}
            for label, v, k in re.findall(
                r"'label':\s*'([^']+)',\s*'value':\s*([\d.]+),\s*'value2':\s*([\d.]+)", r.text
            )
        ]
        if not data:
            raise ValueError("No disease chart returned")
        raw = ROOT / "raw" / f"{key}.html"
        raw.write_text(r.text)
        record.update(
            url=url,
            status="parsed",
            points=len(data),
            sum_cases=sum(x["cases"] for x in data),
            raw_path=str(raw.resolve()),
            sha256=hashlib.sha256(r.content).hexdigest(),
            filters={k: v for k, v in payload.items() if k != "_csrf"},
            data=data,
        )
    except Exception as e:
        record.update(status="failed", error=str(e))
    path.write_text(json.dumps(record, ensure_ascii=False))
    time.sleep(0.15)
    return record


if __name__ == "__main__":
    if (ROOT / "collection_status.json").exists() and json.loads(
        (ROOT / "collection_status.json").read_text()
    ).get("status") == "stopped_source_quality_failure":
        raise SystemExit("Source quality hold: review collection_status.json before resuming")
    (ROOT / "charts").mkdir(parents=True, exist_ok=True)
    (ROOT / "raw").mkdir(exist_ok=True)
    locations = discover_locations()
    # A few UI names share a filter; retain the aliases in locations.json but fetch once.
    locations = list({(x["province"], x["district"]): x for x in locations}.values())
    years = [2023, 2024, 2025, 2026]
    items = [
        (d, y, location) for location in locations for d in ["12MAL", "16SDF"] for y in ["range"]
    ]
    print("Locations", len(locations), "years", years, "charts", len(items), flush=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for f in concurrent.futures.as_completed([pool.submit(fetch, x) for x in items]):
            results.append(f.result())
            if len(results) % 100 == 0:
                print(
                    "Completed",
                    len(results),
                    "parsed",
                    sum(x["status"] == "parsed" for x in results),
                    "nonzero",
                    sum(x.get("sum_cases", 0) > 0 for x in results),
                    flush=True,
                )
    (ROOT / "manifest.json").write_text(
        json.dumps([{k: v for k, v in x.items() if k != "data"} for x in results], indent=2)
    )
    print("Done", len(results), flush=True)

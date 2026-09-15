"""Probe exposed public SKDR filters; distinguish empty feeds from genuine zero surveillance."""

import concurrent.futures
import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://surkarkes.kemkes.go.id"
OUT = Path("../outputs/extensive_mining/dashboards/skdr")


def fetch(item):
    disease, year, province = item
    record = {"disease_code": disease, "year": year, "province": province}
    try:
        s = requests.Session()
        r = s.get(BASE + "/dashboard-skdr-dev", timeout=35)
        r.raise_for_status()
        token = BeautifulSoup(r.text, "html.parser").select_one('meta[name="csrf-token"]')[
            "content"
        ]
        payload = {
            "_csrf": token,
            "kode_penyakit": disease,
            "tahun": year,
            "kd_prop": province,
            "kd_kab": "",
            "jenis_waktu": "1",
            "minggu1": "",
            "minggu2": "",
            "tahun1": "",
            "tahun2": "",
            "multi_penyakit": "0",
        }
        url = BASE + "/index.php/api-skk/grafik-suspek-ibs"
        r = s.post(url, data=payload, timeout=45)
        r.raise_for_status()
        data = [
            {"label": label, "cases": float(v), "klb": float(k)}
            for label, v, k in re.findall(
                r"'label':\s*'([^']+)',\s*'value':\s*([\d.]+),\s*'value2':\s*([\d.]+)", r.text
            )
        ]
        path = OUT / f"{disease}_{year}_{province or 'national'}.html"
        path.write_text(r.text)
        record.update(
            url=url,
            status="parsed" if data else "no_chart_data",
            points=len(data),
            sum_cases=sum(x["cases"] for x in data),
            path=str(path.resolve()),
            data=data,
        )
    except Exception as e:
        record.update(status="failed", error=str(e))
    print({k: v for k, v in record.items() if k not in ["data", "path"]}, flush=True)
    return record


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    items = [
        (d, y, p)
        for d in ["12MAL", "16SDF"]
        for y in ["2023", "2024", "2025", "2026"]
        for p in ["", "33", "91"]
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, items))
    (OUT / "manifest.json").write_text(json.dumps(results, indent=2))

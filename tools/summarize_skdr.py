"""Normalize cached public SKDR charts; keep zero/completeness/calendar caveats explicit."""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path("../outputs/extensive_mining/skdr_nationwide")
if __name__ == "__main__":
    aliases = {}
    for location in json.loads((ROOT / "locations.json").read_text()):
        aliases.setdefault((location["province"], location["district"]), set()).add(
            location["name"]
        )
    records = []
    receipts = []
    for file in sorted((ROOT / "charts").glob("*.json")):
        x = json.loads(file.read_text())
        if x["year"] != "range":
            continue
        receipts.append({k: v for k, v in x.items() if k != "data"})
        if x["status"] != "parsed" or x["year"] != "range":
            continue
        for _i, r in enumerate(x["data"]):
            match = re.fullmatch(r"Mi-(\d+)-(\d+)", r["label"])
            if not match:
                continue
            week, year = int(match[1]), 2000 + int(match[2])
            if x["year"] != "range" and year != int(x["year"]):
                raise ValueError("Chart year mismatch")
            record = {
                "source_family": "skdr_public_dashboard",
                "location_id": "SKDR:" + (x["code"] or "IDN"),
                "location_name": " / ".join(sorted(aliases[(x["province"], x["district"])])),
                "level": x["level"],
                "province_code": x["province"],
                "district_code": x["district"],
                "period": f"{year}-W{week:02}",
                "frequency": "week",
                "disease": "malaria" if x["disease_code"] == "12MAL" else "dengue",
                "metric": "cases",
                "sex": "all",
                "value": r["cases"],
                "case_definition": "confirmed malaria"
                if x["disease_code"] == "12MAL"
                else "suspected dengue",
                "quality": "geography_review"
                if len(aliases[(x["province"], x["district"])]) > 1
                else "empty_year_review"
                if sum(v["cases"] for v in x["data"] if v["label"].endswith(f"-{year % 100:02}"))
                == 0
                else "empty_chart_review"
                if x["sum_cases"] == 0
                else "reporting_completeness_review",
                "note": "Native SKDR chart week; geography codes require a validated crosswalk; zero-filled slots not independently verified; not DBD confirmation counts",
                "source_url": x["url"],
                "raw_path": x["raw_path"],
                "source_locator": f"kode_penyakit={x['disease_code']};year={year};kd_prop={x['province']};kd_kab={x['district']};label={r['label']}",
                "source_sha256": x["sha256"],
                "retrieved_at": x["retrieved_at"],
            }
            record["record_quality_flag"] = record["quality"]
            record["quality"] = "source_validation_failed"
            record["note"] += (
                "; source-wide hold: extreme Aceh Barat 2025-W32 values and missing 2026 reports"
            )
            records.append(record)
    df = pd.DataFrame(records)
    (ROOT / "manifest.json").write_text(json.dumps(receipts, indent=2))
    (ROOT / "observations.json").write_text(json.dumps(records))
    df.to_parquet(ROOT / "observations.parquet", index=False)
    df.to_csv(ROOT / "observations.csv", index=False)
    pd.DataFrame(receipts).to_csv(ROOT / "retrieval_receipts.csv", index=False)
    if len(df):
        report = (
            df.groupby(["level", "disease"])
            .agg(
                observations=("value", "size"),
                locations=("location_id", "nunique"),
                nonzero=("value", lambda v: (v > 0).sum()),
            )
            .reset_index()
        )
        report.to_csv(ROOT / "coverage_summary.csv", index=False)
        print(report.to_string(index=False))
        series = df.groupby(["location_id", "disease", "period"]).size()
        print("Duplicate keys", int((series > 1).sum()))
    print("Charts", len(receipts), "parsed", sum(x["status"] == "parsed" for x in receipts))

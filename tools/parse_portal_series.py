"""Normalize inspected district monthly matrices, preserving source snapshots."""

import csv
import json
import re
from pathlib import Path

import pandas as pd

from dengue_st_diagnostics.extensive_mining import MONTHS, observation

ROOT = Path("../outputs/extensive_mining")
SPECS = {
    "62c0729e-0a73-480e-b34c-5e427b39ee8a": (2022, 10, 7),
    "b66a0ddc-9cfd-4979-9738-f0bec7e54ef2": (2023, 7, 7),
    "40c9bf76-3c4c-401f-962f-8db2056de134": (2023, 10, 2),
    "41b1cd23-731c-4f30-a868-f018e35313ac": (2023, 11, 2),
    "e2c25a15-c7a4-4df0-8980-92e50d1a3ba5": (2024, 1, 2),
}
NAMES = [
    "CILACAP",
    "BANYUMAS",
    "PURBALINGGA",
    "BANJARNEGARA",
    "KEBUMEN",
    "PURWOREJO",
    "WONOSOBO",
    "MAGELANG",
    "BOYOLALI",
    "KLATEN",
    "SUKOHARJO",
    "WONOGIRI",
    "KARANGANYAR",
    "SRAGEN",
    "GROBOGAN",
    "BLORA",
    "REMBANG",
    "PATI",
    "KUDUS",
    "JEPARA",
    "DEMAK",
    "SEMARANG",
    "TEMANGGUNG",
    "KENDAL",
    "BATANG",
    "PEKALONGAN",
    "PEMALANG",
    "TEGAL",
    "BREBES",
    "KOTA MAGELANG",
    "KOTA SURAKARTA",
    "KOTA SALATIGA",
    "KOTA SEMARANG",
    "KOTA PEKALONGAN",
    "KOTA TEGAL",
]
CODES = [str(3300 + i) for i in range(1, 30)] + [str(3370 + i) for i in range(1, 7)]
CODE = dict(zip(NAMES, CODES, strict=False))
if __name__ == "__main__":
    records = []
    audit = []
    for x in json.loads((ROOT / "download_manifest.json").read_text()):
        if x["status"] != "available":
            continue
        rid = x["resource_id"]
        textpath = ROOT / "text" / f"{rid}.txt"
        common = {"url": x["url"], "path": x["path"]}
        if rid in SPECS:
            year, cutoff, start = SPECS[rid]
            text = Path(x["path"]).read_text(encoding="utf-8-sig")
            sep = ";" if text.count(";") > text.count(",") else ","
            for i, row in enumerate(csv.reader(text.splitlines(), delimiter=sep)):
                if len(row) <= start or row[1].strip() not in CODE or not row[0].strip().isdigit():
                    continue
                name = row[1].strip()
                code = CODE[name]
                for month in range(1, cutoff + 1):
                    for metric, offset in [("cases", 0), ("deaths", 1)]:
                        value = row[start + (month - 1) * 2 + offset].strip()
                        if value not in ["", "-"]:
                            # Each snapshot is retained; later reconciliation chooses newest report, never sum revisions.
                            records.append(
                                observation(
                                    f"jateng_report_{year}_{cutoff:02}",
                                    code,
                                    name,
                                    "admin2",
                                    f"{year}-{month:02}",
                                    float(value),
                                    metric=metric,
                                    locator=f"row:{i + 1};month:{month};{metric}",
                                    definition="reported DBD; P=patients, M=deaths",
                                    **common,
                                )
                            )
        elif rid in [
            "5c029a8c-6a51-40cb-97e2-c428e3c0e10e",
            "df6a7845-8392-4241-b6d9-e8a743542a4d",
        ]:
            text = textpath.read_text()
            year = re.search(r"Tahun (20\d\d)", text)[1]
            d = pd.read_excel(x["path"], header=None)
            for i, r in d.iterrows():
                m = MONTHS.get(str(r.iloc[0]).lower().replace("nopember", "november"))
                if not m:
                    continue
                for sex, col in [("male", 1), ("female", 3)]:
                    for metric, offset in [("cases", 0), ("deaths", 1)]:
                        value = pd.to_numeric(r.iloc[col + offset], errors="coerce")
                        if pd.notna(value):
                            records.append(
                                observation(
                                    "belitung_portal",
                                    "1902",
                                    "Kabupaten Belitung",
                                    "admin2",
                                    f"{year}-{m:02}",
                                    value,
                                    metric=metric,
                                    sex=sex,
                                    locator=f"row:{i + 1};col:{col + offset + 1}",
                                    **common,
                                )
                            )
                vals = pd.to_numeric(r.iloc[[1, 3]], errors="coerce")
                if vals.notna().all():
                    records.append(
                        observation(
                            "belitung_portal",
                            "1902",
                            "Kabupaten Belitung",
                            "admin2",
                            f"{year}-{m:02}",
                            vals.sum(),
                            locator=f"row:{i + 1};sum male/female",
                            **common,
                        )
                    )
        elif rid == "968aca1d-b5fb-47ea-a68e-2e6ccbabdf25":
            for i, r in enumerate(csv.reader(Path(x["path"]).read_text().splitlines())):
                if len(r) < 6 or r[0] != "6409":
                    continue
                m = next((n for word, n in MONTHS.items() if word in r[1].lower()), None)
                if m:
                    year = re.search(r"20\d\d", r[1])[0]
                    records.append(
                        observation(
                            "ppu_portal",
                            "6409",
                            "Penajam Paser Utara",
                            "admin2",
                            f"{year}-{m:02}",
                            r[3],
                            disease="malaria",
                            locator=f"row:{i + 1};column4",
                            definition="reported malaria cases found",
                            **common,
                        )
                    )
    # Keep revision history, plus selected latest report per year for one continuous source series.
    d = pd.DataFrame(records)
    latest = (
        d[d.source_family.str.startswith("jateng_report")]
        .sort_values("source_family")
        .drop_duplicates(["location_id", "period", "metric"], keep="last")
    )
    latest["source_family"] = "jateng_latest"
    for row in records:
        if row["source_family"].startswith("jateng_report"):
            row["quality"] = "archived_snapshot"
    records += latest.to_dict(orient="records")
    (ROOT / "portal_observations.json").write_text(json.dumps(records, indent=2))
    print("Rows", len(records), "selected Jateng", len(latest))

"""Extract province cumulative case bars with explicit report cutoffs.

These are cumulative snapshots, not weekly incident counts. Revisions are retained.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pdfplumber

from dengue_st_diagnostics.extensive_mining import observation

ROOT = Path("../outputs/extensive_mining/national_bulletins")


def parse(item):
    rows, audit = [], []
    if item["status"] != "available":
        return rows, audit
    pages = Path(item["text_path"]).read_text().split("\f")
    selected = [
        (i, t)
        for i, t in enumerate(pages)
        if re.match(r"\s*(Suspek Dengue|Malaria Konfirmasi)\s*\n", t)
        and "Berdasarkan Provinsi" in t
    ]
    if not selected:
        return rows, audit
    with pdfplumber.open(item["path"]) as pdf:
        for i, full in selected:
            disease = "malaria" if full.lstrip().startswith("Malaria") else "dengue"
            cutoff = re.search(r"Data\s+s\.?d\.?\s+M[- ]?(\d+)\s*Tahun\s*(20\d{2})", full, re.I)
            if not cutoff:
                audit.append(
                    {"url": item["url"], "page": i + 1, "status": "missing_explicit_cutoff"}
                )
                continue
            week, year = map(int, cutoff.groups())
            if not 1 <= week <= 53:
                continue
            page = pdf.pages[i]
            text = page.crop((0, 0, page.width * 0.44, page.height)).extract_text() or ""
            count = 0
            for line in text.splitlines():
                match = re.match(r"^([A-Z][A-Z ]+)\s+(\d[\d,]*)(?:\s|$)", line)
                if not match:
                    continue
                name, value = match.groups()
                name = name.strip()
                # Province labels are uppercase; exclude headings that contain a year.
                if name.startswith(("DATA", "BERDASARKAN", "KASUS", "TAHUN")):
                    continue
                row = observation(
                    "national_skdr_bulletin",
                    "SKDRPROV:" + name,
                    name,
                    "admin1",
                    f"{year}-W{week:02}",
                    int(value.replace(",", "")),
                    disease=disease,
                    metric="cumulative_cases",
                    frequency="report_week",
                    url=item["url"],
                    path=item["path"],
                    locator=f"page:{i + 1};province bar:{name};cutoff:M{week} {year}",
                    definition="confirmed malaria" if disease == "malaria" else "suspected dengue",
                    quality="cumulative_snapshot",
                    note="Year-to-date cases through explicit printed cutoff; report snapshots may revise prior weeks. Do not interpret as weekly incident cases or sum snapshots.",
                )
                row["source_sha256"] = item["sha256"]
                rows.append(row)
                count += 1
            audit.append(
                {
                    "url": item["url"],
                    "page": i + 1,
                    "status": "parsed",
                    "rows": count,
                    "period": f"{year}-W{week:02}",
                    "disease": disease,
                }
            )
    return rows, audit


if __name__ == "__main__":
    manifest = json.loads((ROOT / "manifest.json").read_text())
    with ThreadPoolExecutor(max_workers=3) as pool:
        parsed = list(pool.map(parse, manifest))
    rows = [r for group, _ in parsed for r in group]
    audit = [r for _, group in parsed for r in group]
    (ROOT / "observations.json").write_text(json.dumps(rows, indent=2))
    (ROOT / "extraction_audit.json").write_text(json.dumps(audit, indent=2))
    print(
        "Province cumulative snapshots",
        len(rows),
        "parsed charts",
        sum(x["status"] == "parsed" for x in audit),
    )

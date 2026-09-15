"""Recover explicitly printed district monthly dengue table cells."""

import json
import re
from pathlib import Path

from dengue_st_diagnostics.extensive_mining import MONTHS, observation

ROOT = Path("../outputs/extensive_mining")
if __name__ == "__main__":
    rows = []
    for item in json.loads((ROOT / "literature/district_papers_manifest.json").read_text()):
        if item["status"] != "available" or "17967" not in item["url"]:
            continue
        text = Path(item["path"]).with_suffix(".txt").read_text()
        table = text.split("Tabel 3. Kasus DBD Perbulan")[1].split("Total/Tahun")[0]
        for match in re.finditer(
            r"^\s*([A-Za-z]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$",
            table,
            re.M,
        ):
            month = 7 if match[1] == "JuIi" else MONTHS.get(match[1].lower())
            if not month:
                continue
            counts = [int(match[i]) for i in range(2, 7)]
            assert sum(counts) == int(match[7])
            for year, value in zip(range(2019, 2024), counts, strict=True):
                row = observation(
                    "tulang_bawang_barat_paper",
                    "1812",
                    "Tulang Bawang Barat",
                    "admin2",
                    f"{year}-{month:02}",
                    value,
                    url=item["url"],
                    path=item["path"],
                    locator=f"Table3;{year};{match[1]}",
                    quality="zero_review" if value == 0 else "observed",
                    note="Table gives zero for November and December in all five years; reporting completeness needs verification. Narrative February 2020 peak says 90, table says 91; preserve table value.",
                )
                row["source_sha256"] = item["sha256"]
                rows.append(row)
    assert len(rows) == 60
    (ROOT / "district_paper_observations.json").write_text(json.dumps(rows, indent=2))
    print("District monthly dengue observations", len(rows))

"""Extract explicitly printed weekly case table cells, never alert counts or absent rows."""

import hashlib
import json
import re
from pathlib import Path

from dengue_st_diagnostics.extensive_mining import observation

ROOT = Path("../outputs/extensive_mining/inhu")


def parse_table(text, item):
    # First heading identifies the current report; some later pages retain stale templates.
    heading = re.search(
        r"MINGGU\s+(?:EPIDEMIOLOGI\s+)?KE[-\u2013 ]*(\d+)\s+TAHUN\s+(20\d{2})", text[:2500], re.I
    )
    if not heading:
        return []
    week, year = map(int, heading.groups())
    if not 1 <= week <= 53:
        raise ValueError("Invalid epidemiological week")
    rows = []
    # Restrict to the table with explicit KASUS / ALERT / KLB column headings.
    for block in text.split("\f"):
        if not re.search(r"KASUS\s+ALERT\s+KLB", block):
            continue
        for match in re.finditer(
            r"\b(Suspek Dengue|Malaria(?: Konfirmasi)?)\s+(\d+)\s+(\d+)\s+(\d+)\s*$",
            block,
            re.M | re.I,
        ):
            label, cases, _alerts, _klb = match.groups()
            row = observation(
                "inhu_skdr_bulletin",
                "1402",
                "Indragiri Hulu",
                "admin2",
                f"{year}-W{week:02}",
                int(cases),
                disease="malaria" if label.lower().startswith("malaria") else "dengue",
                frequency="week",
                url=item["url"],
                path=item["path"],
                locator=f"IBS table;{year}-W{week:02};{label};KASUS",
                definition=label,
                quality="calendar_review",
                note="Explicit weekly case cell. Native Sunday-Saturday epidemiological calendar; missing disease rows are not zero. Later page headings can retain stale template dates.",
            )
            row["source_sha256"] = item["sha256"]
            rows.append(row)
    # Older bulletins print two explicit weekly columns for all monitored diseases.
    first_page = text.split("\f")[0]
    columns = re.search(r"Penyakit\s+M[- .]*(\d+)\s+M[- .]*(\d+)", first_page, re.I)
    if columns and "2 MINGGU TERAKHIR" in first_page.upper():
        weeks = list(map(int, columns.groups()))
        if weeks[-1] == week:
            for match in re.finditer(
                r"\b(Malaria Konfirmasi|Suspek Dengue)\s+(\d+)\s+(\d+)", first_page, re.I
            ):
                for column_week, value in zip(weeks, map(int, match.group(2, 3)), strict=True):
                    column_year = year - 1 if column_week > week else year
                    row = observation(
                        "inhu_skdr_bulletin",
                        "1402",
                        "Indragiri Hulu",
                        "admin2",
                        f"{column_year}-W{column_week:02}",
                        value,
                        disease="malaria" if match[1].lower().startswith("malaria") else "dengue",
                        frequency="week",
                        url=item["url"],
                        path=item["path"],
                        locator=f"page1;two-week IBS table;M{column_week};{match[1]}",
                        definition=match[1],
                        quality="calendar_review",
                        note="Explicit weekly table cell, including printed zero reports. Native epidemiological calendar. Adjacent bulletins can revise prior values.",
                    )
                    row["source_sha256"] = item["sha256"]
                    rows.append(row)
    # Admit facility allocations only when explicit named case counts reconcile to the district total.
    section = re.search(
        r"^\s*\d+\.\s+Suspek Dengue\s*\n(.*?)(?=^\s*\d+\.\s+[A-Za-z]|\Z)", text, re.M | re.S
    )
    totals = {
        r["value"] for r in rows if r["disease"] == "dengue" and r["period"] == f"{year}-W{week:02}"
    }
    names = [
        "Kuala Cenaku",
        "Sipayung",
        "Kampung Besar Kota",
        "Pekan Heran",
        "Pangkalan Kasai",
        "Kilan",
        "Lubuk Kandis",
        "Batang Gansal",
        "Lirik",
        "Air Molek",
        "Sungai Lala",
        "Sungai Parit",
        "Kulim Jaya",
        "Polak Pisang",
        "Rakit Kulim",
        "Peranap",
        "Batang Peranap",
        "Sencano Jaya",
        "Kota Baru",
        "Kota Medan",
    ]
    if section and len(totals) == 1:
        compact = re.sub(r"\s+", " ", section[1])
        pattern = (
            r"(?<!\w)("
            + "|".join(sorted(names, key=len, reverse=True))
            + r")\s+(?:sebanyak\s+)?(\d+)\s+kasus"
        )
        matches = re.findall(pattern, compact, re.I)
        if (
            matches
            and len({n.lower() for n, _ in matches}) == len(matches)
            and sum(int(v) for _, v in matches) == next(iter(totals))
        ):
            for name, value in matches:
                name = name.title()
                row = observation(
                    "inhu_skdr_bulletin",
                    f"1402:puskesmas:{name}",
                    name,
                    "facility",
                    f"{year}-W{week:02}",
                    int(value),
                    frequency="week",
                    url=item["url"],
                    path=item["path"],
                    locator="Suspek Dengue narrative;explicit facility case count reconciled to district",
                    definition="suspected dengue",
                    quality="narrative_review",
                    note="Explicit facility cases sum to printed district weekly total; retained for visual narrative verification.",
                )
                row["source_sha256"] = item["sha256"]
                rows.append(row)
    return rows


if __name__ == "__main__":
    rows, audit = [], []
    for item in json.loads((ROOT / "pdf_manifest.json").read_text()):
        if item["status"] != "available":
            continue
        parsed = parse_table(Path(item["text_path"]).read_text(), item)
        rows.extend(parsed)
        audit.append({"url": item["url"], "rows": len(parsed), "sha256": item["sha256"]})
    # Visually inspected M15 2026 narrative explicitly assigns 3 and 2 cases to these facilities.
    item = next(
        (
            x
            for x in json.loads((ROOT / "pdf_manifest.json").read_text())
            if "fb2ef92702f28e66bb.pdf" in x.get("path", "")
        ),
        None,
    )
    if item:
        for facility, value in [("Pangkalan Kasai", 3), ("Kilan", 2)]:
            row = observation(
                "inhu_skdr_bulletin",
                f"1402:puskesmas:{facility}",
                facility,
                "facility",
                "2026-W15",
                value,
                frequency="week",
                url=item["url"],
                path=item["path"],
                locator="page5;Suspek Dengue narrative;facility cases",
                definition="suspected dengue",
                quality="calendar_review",
                note="Explicit current-week narrative count; native epidemiological week, not ISO date assumption.",
            )
            row["source_sha256"] = hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
            rows.append(row)
    (ROOT / "observations.json").write_text(json.dumps(rows, indent=2))
    (ROOT / "extraction_audit.json").write_text(json.dumps(audit, indent=2))
    print(
        "Weekly observations",
        len(rows),
        "PDFs with explicit target case cells",
        sum(x["rows"] > 0 for x in audit),
        "PDFs inspected",
        len(audit),
    )

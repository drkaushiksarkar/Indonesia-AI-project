"""Compare new monthly case candidates with existing native lake; do not count aliases twice."""

from pathlib import Path

import pandas as pd
import psycopg

ROOT = Path("../outputs/extensive_mining")
ALIASES = {
    "BABEL": "KEPULAUAN BANGKA BELITUNG",
    "BANGKA BELITUNG": "KEPULAUAN BANGKA BELITUNG",
    "D.I YOGYA": "DAERAH ISTIMEWA YOGYAKARTA",
    "D.I. YOGYAKARTA": "DAERAH ISTIMEWA YOGYAKARTA",
    "KALIMANTAN SELATA": "KALIMANTAN SELATAN",
    "SULAWESI TENGGAR": "SULAWESI TENGGARA",
    "KEP.RIAU": "KEPULAUAN RIAU",
    "KEPRI": "KEPULAUAN RIAU",
    "NUSA TENGGARA TI": "NUSA TENGGARA TIMUR",
    "NUSA TENGGARA BA": "NUSA TENGGARA BARAT",
}


def norm(s):
    s = str(s or "").strip().upper().replace("SUMATERA", "SUMATRA")
    return ALIASES.get(s, s)


if __name__ == "__main__":
    with psycopg.connect(dbname="indonesia_vector_lake") as c:
        existing = c.execute(
            "select admin_1,admin_2,facility_name,period_start,disease,value,spatial_resolution from gold.native_observation where lower(temporal_resolution)='month' and metric in ('cases','confirmed_cases','malaria_cases')"
        ).fetchall()
    lookup = {}
    for a1, a2, fac, period, disease, value, level in existing:
        loc = fac if level == "facility" else a1 if level == "adm1" else a2
        level = {"adm1": "admin1", "adm2": "admin2"}.get(level, level)
        key = (norm(loc), period.strftime("%Y-%m"), disease, level)
        lookup.setdefault(key, set()).add(value)
    new = pd.read_parquet(ROOT / "normalized/observations.parquet")
    new = new[
        (new.metric == "cases")
        & (new.sex == "all")
        & (new.frequency == "month")
        & ~new.source_family.str.startswith("jateng_report")
    ]
    stats = []
    for r in new.to_dict(orient="records"):
        values = lookup.get((norm(r["location_name"]), r["period"], r["disease"], r["level"]))
        r["baseline_match"] = (
            "same_count_present"
            if values and r["value"] in values
            else "different_count_present"
            if values
            else "no_exact_location_period_match"
        )
        stats.append(r)
    df = pd.DataFrame(stats)
    df.to_csv(ROOT / "normalized/baseline_comparison.csv", index=False)
    summary = df.groupby(["source_family", "baseline_match"]).size().unstack(fill_value=0)
    summary.to_csv(ROOT / "normalized/baseline_comparison_summary.csv")
    print(summary.to_string())

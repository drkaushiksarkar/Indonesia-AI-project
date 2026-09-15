"""Evidence-preserving normalization of explicitly inspected public time-series sources.

No spatial disaggregation, missing-to-zero conversion, or cross-source summation.
Weekly dashboard labels remain native year/week until its calendar is documented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "januari",
            "februari",
            "maret",
            "april",
            "mei",
            "juni",
            "juli",
            "agustus",
            "september",
            "oktober",
            "november",
            "desember",
        ],
        1,
    )
}
RESEARCH = "https://github.com/mlgh-sg/dengue-climate-indonesia"
COMMIT = "c7dd60c9a0547723a6f0b156cee1a86b5604fbeb"
KEY = ["source_family", "location_id", "disease", "metric", "sex", "frequency", "period"]


def observation(
    source,
    location,
    name,
    level,
    period,
    value,
    *,
    metric="cases",
    disease="dengue",
    sex="all",
    url="",
    path="",
    locator="",
    definition="reported DBD cases",
    quality="observed",
    note="",
    frequency="month",
):
    value = float(value)
    if not np.isfinite(value) or value < 0 or value != int(value):
        raise ValueError(f"Invalid count at {locator}: {value}")
    return {
        "source_family": source,
        "location_id": location,
        "location_name": name,
        "level": level,
        "period": period,
        "frequency": frequency,
        "disease": disease,
        "metric": metric,
        "sex": sex,
        "value": value,
        "case_definition": definition,
        "quality": quality,
        "note": note,
        "source_url": url,
        "raw_path": str(path),
        "source_locator": locator,
    }


def semarang_records(payload, item):
    if not payload.get("status"):
        return []
    year = int(item["year"])
    weekly = item["variable"] == "dbd_mingguan"
    frequency = "week" if weekly else "month"
    series = payload.get("data", [])
    labels = payload.get("mingguan") if weekly else None
    lengths = {len(s["data"]) for s in series}
    if len(lengths) != 1:
        raise ValueError("Unequal dashboard series lengths")
    length = next(iter(lengths), 0)
    if weekly and (labels is None or len(labels) != length or len(set(labels)) != length):
        raise ValueError("Weekly values require unique explicit week labels")
    if not weekly and length > 12:
        raise ValueError("Too many monthly values")
    labels = labels if weekly else range(1, length + 1)
    rows = []
    malaria = item["variable"] == "kasus_malaria"
    for s in series:
        label = s["name"].lower()
        metric = "deaths" if "meninggal" in label else "cases"
        sex = "male" if "laki" in label else "female" if "perempuan" in label else "all"
        for i, (period, value) in enumerate(zip(labels, s["data"], strict=True)):
            if value is None:
                continue
            rows.append(
                observation(
                    "semarang_dashboard",
                    "3374",
                    "Kota Semarang",
                    "admin2",
                    f"{year}-W{int(period):02}" if weekly else f"{year}-{int(period):02}",
                    value,
                    metric=metric,
                    disease="malaria" if malaria else "dengue",
                    sex=sex,
                    url=item["url"],
                    path=item["path"],
                    locator=f"data/{s['name']}/{i}",
                    frequency=frequency,
                    definition=payload["title"],
                    quality="calendar_review"
                    if weekly
                    else "provisional"
                    if year >= 2025
                    else "observed",
                    note="Native week labels; calendar mapping unresolved"
                    if weekly
                    else "UI maps positions to Jan-Dec; reporting completeness unverified"
                    if year >= 2025
                    else "",
                )
            )
    # Both sexes are disjoint. Derive all-sex counts only where both are observed.
    frame = pd.DataFrame(rows)
    if not frame.empty and set(frame.sex) == {"male", "female"}:
        for _, group in frame.groupby(["period", "metric"]):
            if len(group) == 2:
                record = group.iloc[0].to_dict()
                record.update(
                    sex="all", value=float(group.value.sum()), source_locator="sum(male,female)"
                )
                rows.append(record)
    return rows


def normalize_sources(root):
    rows, audit = [], []
    casepath = root / "research/dengue_data_admin1_indonesia.csv"
    provinces = pd.read_csv(casepath)
    for idx, r in provinces.iterrows():
        code = str(int(r.idadmin1))
        # All province boundaries and zero semantics require review before model admission.
        quality = (
            "boundary_review"
            if code in {"65", "91", "94"}
            else "zero_review"
            if r.cases == 0
            else "observed"
        )
        rows.append(
            observation(
                "research_provincial",
                code,
                r.admin1,
                "admin0" if r.admin1 == "INDONESIA" else "admin1",
                f"{int(r.year)}-{int(r.month):02}",
                r.cases,
                url=f"{RESEARCH}/blob/{COMMIT}/data/dengue_data_admin1_indonesia.rds",
                path=casepath,
                locator=f"row:{idx + 1}",
                quality=quality,
                note="Source uses historical 34-province geography; zeros may encode missing reports",
                definition="research repository dengue case count",
            )
        )
    for item in json.loads((root / "semarang/manifest.json").read_text()):
        if item["status"] == "available":
            rows.extend(semarang_records(json.loads(Path(item["path"]).read_text()), item))
    manifests = json.loads((root / "download_manifest.json").read_text())
    for item in manifests:
        if item["status"] != "available":
            continue
        title, path = item["title"], Path(item["path"])
        rid = item["resource_id"]
        textpath = root / "text" / f"{rid}.txt"
        text = textpath.read_text() if textpath.exists() else ""
        count_before = len(rows)
        common = {"url": item["url"], "path": path}
        try:
            if rid in {
                "223406fa-c041-4cf8-9b62-bc0293478d7b",
                "3c9eaf6a-95c8-45d9-bc23-20945a0bac3b",
            }:
                table = pd.read_excel(path, header=None)
                year = int(re.search(r"TAHUN\s+(20\d\d)", text).group(1))
                disease = "malaria" if "malaria" in title.lower() else "dengue"
                for idx, r in table.iterrows():
                    month = MONTHS.get(str(r.iloc[0]).strip().lower())
                    if not month:
                        continue
                    vals = pd.to_numeric(r.iloc[1:6], errors="coerce")
                    if vals.isna().any() or vals.iloc[:4].sum() != vals.iloc[4]:
                        raise ValueError("Subunit/total reconciliation failed")
                    for col, name in enumerate(
                        ["Mojolangu", "Tunjungsekar", "Tunggulwulung", "Tasikmadu"]
                    ):
                        rows.append(
                            observation(
                                "malang_mined",
                                f"3573:kel:{name}",
                                name,
                                "kelurahan",
                                f"{year}-{month:02}",
                                vals.iloc[col],
                                disease=disease,
                                locator=f"row:{idx + 1};column:{col + 2}",
                                **common,
                            )
                        )
                    rows.append(
                        observation(
                            "malang_mined",
                            "3573:puskesmas:Mojolangu",
                            "Mojolangu",
                            "facility",
                            f"{year}-{month:02}",
                            vals.iloc[4],
                            disease=disease,
                            locator=f"row:{idx + 1};total",
                            **common,
                        )
                    )
            elif rid == "937dcdbd-b837-47fe-86f0-529675b68fa6":
                table = pd.read_excel(path, header=None)
                assert "TAHUN 2022" in text and "KASUS SEMBUH" in text and "KASUS MD" in text
                for month in range(1, 13):
                    recoveries = pd.to_numeric(table.iloc[5:9, 2 * month], errors="coerce")
                    deaths = pd.to_numeric(table.iloc[5:9, 2 * month + 1], errors="coerce")
                    if recoveries.isna().any() or deaths.isna().any():
                        raise ValueError("Incomplete outcome partition")
                    for metric, values in [
                        ("recovered_cases", recoveries),
                        ("deaths", deaths),
                        ("cases", recoveries + deaths),
                    ]:
                        for offset, name in enumerate(
                            ["Arjowinangun", "Bumiayu", "Mergosono", "Tlogowaru"]
                        ):
                            rows.append(
                                observation(
                                    "malang_mined",
                                    f"3573:kel:{name}",
                                    name,
                                    "kelurahan",
                                    f"2022-{month:02}",
                                    values.iloc[offset],
                                    metric=metric,
                                    locator=f"row:{offset + 6};month:{month}",
                                    definition="DBD recovered plus died"
                                    if metric == "cases"
                                    else metric,
                                    **common,
                                )
                            )
                        rows.append(
                            observation(
                                "malang_mined",
                                "3573:puskesmas:Arjowinangun",
                                "Arjowinangun",
                                "facility",
                                f"2022-{month:02}",
                                values.sum(),
                                metric=metric,
                                locator=f"sum rows6:9;month:{month}",
                                definition="DBD recovered plus died"
                                if metric == "cases"
                                else metric,
                                **common,
                            )
                        )
            elif title.startswith("DATA PENDERITA DBD MENURUT WILAYAH RW PUSKESMAS PANDANWANGI "):
                month = next((n for m, n in MONTHS.items() if m in title.lower()), None)
                if month:
                    year = re.search(r"20\d\d", title).group()
                    # Layout verified against PDF: two explicit subunit totals above Jumlah, total below.
                    match = re.search(r"\n\s*(\d+)\s+(\d+)\s*\nJumlah\s*\n\s*(\d+)", text)
                    if not match:
                        raise ValueError("Explicit totals not found")
                    arjosari, pandanwangi, total = map(int, match.groups())
                    if arjosari + pandanwangi != total:
                        raise ValueError("Subunit/total mismatch")
                    for name, value in [("Arjosari", arjosari), ("Pandanwangi", pandanwangi)]:
                        rows.append(
                            observation(
                                "malang_mined",
                                f"3573:kel:{name}",
                                name,
                                "kelurahan",
                                f"{year}-{month:02}",
                                value,
                                locator="page1;Jumlah subunit",
                                **common,
                            )
                        )
                    rows.append(
                        observation(
                            "malang_mined",
                            "3573:puskesmas:Pandanwangi",
                            "Pandanwangi",
                            "facility",
                            f"{year}-{month:02}",
                            total,
                            locator="page1;Jumlah total",
                            **common,
                        )
                    )
            elif re.search(r"(Data|Grafik) Kasus DBD Kel", title, re.I) and "KASUS" in text:
                name = next(
                    (
                        n
                        for n in ["Arjowinangun", "Bumiayu", "Mergosono", "Tlogowaru"]
                        if n.lower() in title.lower()
                    ),
                    None,
                )
                year = re.search(r"20\d\d", title)
                if name and year:
                    for match in re.finditer(r"^\s*\d+\s+([A-Z]+)\s+(\d+)\s+(\d+)\b", text, re.M):
                        month = MONTHS.get(match[1].lower())
                        if not month:
                            continue
                        for metric, value in [
                            ("recovered_cases", int(match[2])),
                            ("deaths", int(match[3])),
                            ("cases", int(match[2]) + int(match[3])),
                        ]:
                            rows.append(
                                observation(
                                    "malang_mined",
                                    f"3573:kel:{name}",
                                    name,
                                    "kelurahan",
                                    f"{year[0]}-{month:02}",
                                    value,
                                    metric=metric,
                                    locator=f"page1;{match[1]}",
                                    definition="DBD recovered plus died"
                                    if metric == "cases"
                                    else metric,
                                    **common,
                                )
                            )
        except Exception as e:
            del rows[count_before:]
            audit.append({"resource_id": rid, "status": "review", "reason": str(e)})
        else:
            if len(rows) > count_before:
                audit.append(
                    {
                        "resource_id": rid,
                        "status": "parsed",
                        "observations": len(rows) - count_before,
                    }
                )
    # Derive Arjowinangun totals only where all four explicit kelurahan values exist.
    frame = pd.DataFrame(rows)
    selected = frame[
        (frame.source_family == "malang_mined")
        & (frame.level == "kelurahan")
        & frame.location_name.isin(["Arjowinangun", "Bumiayu", "Mergosono", "Tlogowaru"])
        & ~frame.period.str.startswith("2022")
    ]
    for (_period, _metric), group in selected.groupby(["period", "metric"]):
        if len(group) == 4 and group.location_id.nunique() == 4:
            r = group.iloc[0].to_dict()
            r.update(
                location_id="3573:puskesmas:Arjowinangun",
                location_name="Arjowinangun",
                level="facility",
                value=float(group.value.sum()),
                source_locator="sum four kelurahan",
                source_url=json.dumps(group.source_url.tolist()),
                raw_path=json.dumps(group.raw_path.tolist()),
            )
            rows.append(r)
    study_path = root / "primary/Figure 5_6_DengueCaseData.xlsx"
    study = pd.read_excel(study_path, sheet_name="Data")
    for idx, r in study.iterrows():
        for arm in ["Intervention", "Control"]:
            for metric, col in [("cases", "DHF_Cases"), ("df_dhf_cases", "DFDHF_Cases")]:
                value = r[f"{col}_{arm}"]
                if pd.notna(value):
                    rows.append(
                        observation(
                            "yogyakarta_wolbachia",
                            f"YOG:study:{arm}",
                            f"Yogyakarta {arm}",
                            "study_area",
                            pd.Timestamp(r["Month"]).strftime("%Y-%m"),
                            value,
                            metric=metric,
                            url="https://doi.org/10.6084/m9.figshare.12199688",
                            path=study_path,
                            locator=f"row:{idx + 2};{col}_{arm}",
                            definition="DHF" if metric == "cases" else "DF+DHF",
                            note="Study catchment; not whole municipality. Intervention changes transmission.",
                        )
                    )
    if (root / "paper_observations.json").exists():
        rows.extend(json.loads((root / "paper_observations.json").read_text()))
    if (root / "portal_observations.json").exists():
        rows.extend(json.loads((root / "portal_observations.json").read_text()))
    for extra in [
        "district_paper_observations.json",
        "inhu/observations.json",
        "national_bulletins/observations.json",
        "skdr_nationwide/observations.json",
    ]:
        if (root / extra).exists():
            rows.extend(json.loads((root / extra).read_text()))
    return pd.DataFrame(rows), audit


def deduplicate(frame):
    conflicts = frame.groupby(KEY, dropna=False).value.nunique()
    conflict_keys = conflicts[conflicts > 1].reset_index()[KEY]
    if not conflict_keys.empty:
        frame = frame.merge(conflict_keys.assign(conflict=True), on=KEY, how="left")
        frame.loc[frame.conflict.eq(True), "quality"] = "conflict"
        frame = frame.drop(columns="conflict")
    return frame.drop_duplicates([*KEY, "value"]), conflict_keys


def monthly_features(frame):
    """Strict calendar lags for case-count histories only."""
    safe = frame[
        (frame.frequency == "month")
        & (frame.metric == "cases")
        & (frame.sex == "all")
        & (frame.quality == "observed")
        & frame.level.isin(["admin1", "admin2", "facility"])
    ].copy()
    result = []
    if safe.empty:
        for lag in [1, 2, 3, 6, 12]:
            safe[f"cases_lag{lag}"] = pd.Series(dtype=float)
        safe["cases_rolling3_prior"] = pd.Series(dtype=float)
        safe["target_history_complete"] = pd.Series(dtype=bool)
        return safe
    for _, g in safe.groupby(["source_family", "location_id", "disease"]):
        if g.period.duplicated().any():
            continue
        g = g.set_index(pd.PeriodIndex(g.period, freq="M")).sort_index()
        calendar = pd.period_range(g.index.min(), g.index.max(), freq="M")
        values = g.value.reindex(calendar)
        for lag in [1, 2, 3, 6, 12]:
            g[f"cases_lag{lag}"] = values.shift(lag).reindex(g.index).to_numpy()
        g["cases_rolling3_prior"] = (
            values.shift(1).rolling(3, min_periods=3).mean().reindex(g.index).to_numpy()
        )
        result.append(g.reset_index(drop=True))
    features = pd.concat(result, ignore_index=True)
    lagcols = [f"cases_lag{lag}" for lag in [1, 2, 3, 6, 12]] + ["cases_rolling3_prior"]
    features["target_history_complete"] = features[lagcols].notna().all(axis=1)
    return features


def readiness(frame):
    rows = []
    target = frame[(frame.metric == "cases") & (frame.sex == "all")]
    for keys, g in target.groupby(
        ["source_family", "location_id", "location_name", "level", "disease", "frequency"]
    ):
        periods = sorted(set(g.period))
        longest = 0
        if keys[-1] == "month":
            p = pd.PeriodIndex(periods, freq="M")
            expected = len(pd.period_range(p.min(), p.max(), freq="M"))
            streak = 0
            previous = None
            for x in p:
                streak = (
                    streak + 1 if previous is not None and x.ordinal - previous.ordinal == 1 else 1
                )
                longest = max(longest, streak)
                previous = x
        else:
            # Year/week is native, not ISO. No cross-year calendar assumptions.
            expected = None
        rows.append(
            dict(
                zip(
                    [
                        "source_family",
                        "location_id",
                        "location_name",
                        "level",
                        "disease",
                        "frequency",
                    ],
                    keys,
                    strict=True,
                )
            )
            | {
                "periods": len(periods),
                "first": periods[0],
                "last": periods[-1],
                "expected": expected,
                "longest_contiguous_months": longest,
                "review_rows": int((g.quality != "observed").sum()),
                "nonzero_periods": int((g.value > 0).sum()),
                "history_screen": "36_months_available"
                if longest >= 36
                else "short_or_calendar_review",
            }
        )
    return pd.DataFrame(rows)


def publish(root, frames, dbname):
    """Idempotent dedicated tables; leave existing historical native observations untouched."""
    with psycopg.connect(dbname=dbname) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS bronze.extensive_mining_resource (resource_key text PRIMARY KEY, payload jsonb NOT NULL)"
        )
        resources = json.loads((root / "download_manifest.json").read_text()) + json.loads(
            (root / "semarang/manifest.json").read_text()
        )
        for manifest in [
            "literature/manifest.json",
            "literature/additional_manifest.json",
            "literature/district_papers_manifest.json",
            "national_bulletins/manifest.json",
            "reference_cases_manifest.json",
            "inhu/pdf_manifest.json",
            "skdr_nationwide/manifest.json",
        ]:
            if (root / manifest).exists():
                entries = json.loads((root / manifest).read_text())
                if isinstance(entries, list):
                    resources.extend(entries)
        for path in (root / "research").glob("dengue*.rds"):
            resources.append(
                {
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "url": f"{RESEARCH}/blob/{COMMIT}/data/{path.name}",
                    "status": "available",
                }
            )
        for r in resources:
            key = hashlib.sha256((r.get("url", "") + r.get("sha256", "")).encode()).hexdigest()
            conn.execute(
                "INSERT INTO bronze.extensive_mining_resource VALUES (%s,%s) ON CONFLICT(resource_key) DO UPDATE SET payload=excluded.payload",
                (key, Jsonb(r)),
            )
        for name, frame in frames.items():
            # Names are internal constants supplied below, never external SQL input.
            from psycopg import sql

            table = sql.Identifier(*name.split("."))
            conn.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {} (record_id text PRIMARY KEY, payload jsonb NOT NULL)"
                ).format(table)
            )
            conn.execute(sql.SQL("DELETE FROM {}").format(table))
            records = json.loads(frame.to_json(orient="records"))
            with conn.cursor().copy(
                sql.SQL("COPY {} (record_id,payload) FROM STDIN").format(table)
            ) as copy:
                for record in records:
                    encoded = json.dumps(record, sort_keys=True)
                    copy.write_row((hashlib.sha256(encoded.encode()).hexdigest(), Jsonb(record)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("../outputs/extensive_mining"))
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--database", default="indonesia_vector_lake")
    args = parser.parse_args()
    out = args.root / "normalized"
    out.mkdir(exist_ok=True)
    observations, audit = normalize_sources(args.root)
    hashes = {}
    for raw_path in observations.raw_path.dropna().unique():
        path = Path(raw_path)
        if path.is_file():
            hashes[raw_path] = hashlib.sha256(path.read_bytes()).hexdigest()
    observations["source_sha256"] = observations.raw_path.map(hashes)
    observations, conflicts = deduplicate(observations)
    comparison_path = out / "baseline_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path, dtype={"location_id": str})
        comparison = comparison[[*KEY, "value", "baseline_match"]].drop_duplicates()
        observations = observations.merge(
            comparison, on=[*KEY, "value"], how="left", validate="many_to_one"
        )
        held = observations.baseline_match.eq("different_count_present") & observations.quality.eq(
            "observed"
        )
        observations.loc[held, "quality"] = "cross_source_review"
    features = monthly_features(observations)
    coverage = readiness(observations)
    for name, frame in [
        ("observations", observations),
        ("monthly_features", features),
        ("series_readiness", coverage),
        ("conflicts", conflicts),
    ]:
        frame.to_parquet(out / f"{name}.parquet", index=False)
        frame.to_csv(out / f"{name}.csv", index=False)
    (out / "extraction_audit.json").write_text(json.dumps(audit, indent=2))
    summary = {
        "observations": len(observations),
        "feature_rows": len(features),
        "complete_history_rows": int(features.target_history_complete.sum()),
        "conflict_keys": len(conflicts),
        "series_by_level": coverage.groupby("level").size().to_dict(),
        "observed_by_quality": observations.groupby("quality").size().to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if args.publish:
        publish(
            args.root,
            {
                "silver.extensive_mining_observation": observations,
                "gold.extensive_mining_features": features[features.target_history_complete],
                "gold.extensive_mining_readiness": coverage,
            },
            args.database,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

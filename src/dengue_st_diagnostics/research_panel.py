"""Recover an explicitly identified aggregate research panel from preserved bronze cells.

The gold panel is for retrospective evaluation. Publication dates and geographic
boundary harmonization are not established; it is not an operational forecast feed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

SOURCE_RECORD = "20034615"
VERSION = "zenodo_province_panel_1.0.0"
# Exact source headers; fail closed if the source changes.
FIELDS = {
    "Laki-laki": ("male_population", "people"),
    "Perempuan": ("female_population", "people"),
    "Total Penduduk": ("population", "people"),
    "Luas Wilayah (Km²)": ("area", "km2"),
    "Kepadatan Penduduk (Jiwa per Km²)": ("population_density", "people/km2"),
    "Jumlah Puskesmas": ("puskesmas", "facilities"),
    "Jumlah Kasus": ("cases", "cases"),
    "Incidence Rate per 100.000 Penduduk": ("incidence_rate", "per_100000_people"),
    "Jumlah Kasus Meninggal": ("deaths", "deaths"),
    "Case Fatality Rate (%)": ("case_fatality_rate", "percent"),
    "Lingkungan Tatanan Sehat (%)": ("healthy_environment", "percent"),
    "Sanitasi Layak (%)": ("adequate_sanitation", "percent"),
    "Kualitas kesehatan lingkungan (%)": ("environmental_health_quality", "percent"),
    "Persentase Rumah Tangga Kumuh (%)": ("slum_households", "percent"),
    "TFS Sesuai Standar (%)": ("tfs_source_defined", "percent"),
    "Kelayakan Sumber Air Minum (%)": ("adequate_drinking_water", "percent"),
    "Indeks Gini": ("gini", "source_index"),
    "Indeks Pembangunan Manusia": ("human_development_index", "source_index"),
    "Total Rumah Sakit": ("hospitals", "facilities"),
    "Persentase Penduduk Miskin": ("poverty", "percent"),
    "Persentase PVT": ("pvt_source_defined", "percent"),
}
# Unknown abbreviations and contemporaneous outcome-derived rates are excluded.
PREDICTORS = {
    "population",
    "area",
    "population_density",
    "puskesmas",
    "hospitals",
    "adequate_sanitation",
    "adequate_drinking_water",
    "poverty",
    "gini",
    "human_development_index",
    "cases",
    "deaths",
}
ALIASES = {"Kep. Bangka Belitung": "Kepulauan Bangka Belitung"}


def fetch_cells(connection: Any) -> list[tuple[int, dict[str, Any]]]:
    return connection.execute(
        """select bronze_record_id, payload from bronze.record
        where dataset_id = (select dataset_id from metadata.dataset
            where dataset_name='public_tabular_cells'
            order by loaded_at desc, dataset_id desc limit 1)
        and payload->>'source'='zenodo' and payload->>'dataset_id'=%s
        and payload->>'sheet'='csv'
        order by (payload->>'row_number')::integer,
                 (payload->>'column_number')::integer""",
        (SOURCE_RECORD,),
    ).fetchall()


def extract(cells: list[tuple[int, dict[str, Any]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix: dict[int, dict[int, tuple[int, dict[str, Any]]]] = defaultdict(dict)
    for cell_id, payload in cells:
        row, column = int(payload["row_number"]), int(payload["column_number"])
        if column in matrix[row]:
            raise ValueError(f"Duplicate source coordinate: {row}, {column}")
        matrix[row][column] = (cell_id, payload)
    headers = {c: p["value"].strip() for c, (_, p) in matrix.get(1, {}).items()}
    if set(headers.values()) != {"Tahun", "Provinsi", *FIELDS} or len(headers) != 23:
        raise ValueError("Unexpected panel schema; manual source review required")
    observations, problems = [], []
    seen = set()
    for row_number, source_row in sorted(matrix.items()):
        if row_number == 1:
            continue
        values = {headers[c]: p["value"].strip() for c, (_, p) in source_row.items()}
        year = int(values["Tahun"])
        if not 2017 <= year <= 2024:
            raise ValueError("Unexpected source year")
        province_raw = values["Provinsi"]
        province = ALIASES.get(province_raw, province_raw)
        if not province or (province, year) in seen:
            raise ValueError(f"Invalid or duplicate province-year: {province}, {year}")
        seen.add((province, year))
        for column, header in headers.items():
            if header not in FIELDS:
                continue
            metric, unit = FIELDS[header]
            cell_id, payload = source_row.get(column, (None, {}))
            raw = values.get(header, "")
            number = pd.to_numeric(raw, errors="coerce")
            valid = pd.notna(number) and float("-inf") < number < float("inf") and number >= 0
            if unit == "percent" and valid:
                valid = number <= 100
            if unit in {"people", "facilities", "cases", "deaths"} and valid:
                valid = float(number).is_integer()
            if not valid:
                problems.append(
                    {
                        "province": province,
                        "year": year,
                        "metric": metric,
                        "raw_value": raw,
                        "reason": "missing_or_out_of_range",
                    }
                )
            key = f"zenodo:{SOURCE_RECORD}:{province}:{year}:{metric}"
            observations.append(
                {
                    "observation_key": hashlib.sha256(key.encode()).hexdigest(),
                    "source_record": SOURCE_RECORD,
                    "province": province,
                    "source_province": province_raw,
                    "year": year,
                    "metric": metric,
                    "unit": unit,
                    "value": float(number) if valid else None,
                    "quality_status": "valid" if valid else "review_required",
                    "source_header": header,
                    "source_row": row_number,
                    "source_column": column,
                    "bronze_record_id": cell_id,
                    "source_url": payload.get("source_url", ""),
                    "parser_version": VERSION,
                }
            )
    frame = pd.DataFrame(observations)
    wide = frame.pivot(index=["province", "year"], columns="metric", values="value")
    for (province, year), values in wide.iterrows():
        checks = {
            "population_components_disagree": abs(
                values.male_population + values.female_population - values.population
            )
            > 1,
            "incidence_disagrees_with_cases_population": values.population > 0
            and abs(values.cases / values.population * 100000 - values.incidence_rate) > 1,
            "fatality_rate_disagrees_with_cases_deaths": values.cases > 0
            and abs(values.deaths / values.cases * 100 - values.case_fatality_rate) > 0.1,
            "deaths_exceed_cases": values.deaths > values.cases,
        }
        for reason, failed in checks.items():
            if failed:
                problems.append(
                    {
                        "province": province,
                        "year": year,
                        "metric": "row_consistency",
                        "raw_value": "",
                        "reason": reason,
                    }
                )
                frame.loc[frame.province.eq(province) & frame.year.eq(year), "quality_status"] = (
                    "review_required"
                )
    return frame, pd.DataFrame(problems)


def build_features(observations: pd.DataFrame) -> pd.DataFrame:
    valid = observations[observations.quality_status.eq("valid")]
    wide = valid.pivot(index=["province", "year"], columns="metric", values="value")
    rows = []
    for (province, year), current in wide.iterrows():
        if (province, year - 1) not in wide.index or pd.isna(current.get("cases")):
            continue
        previous = wide.loc[(province, year - 1)]
        predictors = {
            f"{key}_lag1": float(previous[key])
            for key in sorted(PREDICTORS)
            if key in previous and pd.notna(previous[key])
        }
        # Complete-case features only; never silently impute an absent year or value.
        if len(predictors) != len(PREDICTORS):
            continue
        rows.append(
            {
                "source_record": SOURCE_RECORD,
                "province": province,
                "target_year": int(year),
                "predictor_year": int(year - 1),
                "target_cases": float(current["cases"]),
                "predictors": predictors,
                "readiness": "retrospective_only",
                "boundary_status": "source_geography_not_harmonized",
                "availability_status": "publication_dates_not_verified",
            }
        )
    return pd.DataFrame(rows)


def publish(connection: Any, observations: pd.DataFrame, features: pd.DataFrame) -> None:
    connection.execute("""create table if not exists silver.research_panel_observation (
        observation_key text primary key, bronze_record_id bigint references bronze.record,
        source_record text not null, province text not null, year integer not null,
        metric text not null, value double precision, unit text not null,
        quality_status text not null, provenance jsonb not null)""")
    connection.execute("""create table if not exists gold.research_panel_features (
        source_record text not null, province text not null, target_year integer not null,
        predictor_year integer not null, target_cases double precision not null,
        predictors jsonb not null, readiness text not null, provenance jsonb not null,
        primary key(source_record,province,target_year),
        check(predictor_year = target_year - 1))""")
    # Refresh only this named source within one transaction, including removed rows.
    connection.execute(
        "delete from gold.research_panel_features where source_record=%s", (SOURCE_RECORD,)
    )
    connection.execute(
        "delete from silver.research_panel_observation where source_record=%s", (SOURCE_RECORD,)
    )
    for row in json.loads(observations.to_json(orient="records")):
        connection.execute(
            """insert into silver.research_panel_observation values
            (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                *[
                    row[k]
                    for k in [
                        "observation_key",
                        "bronze_record_id",
                        "source_record",
                        "province",
                        "year",
                        "metric",
                        "value",
                        "unit",
                        "quality_status",
                    ]
                ],
                Jsonb(row),
            ),
        )
    for row in features.to_dict(orient="records"):
        connection.execute(
            """insert into gold.research_panel_features values
            (%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                row["source_record"],
                row["province"],
                row["target_year"],
                row["predictor_year"],
                row["target_cases"],
                Jsonb(row["predictors"]),
                row["readiness"],
                Jsonb(row),
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="indonesia_vector_lake")
    parser.add_argument("--output", type=Path, default=Path("outputs/research_panel"))
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with psycopg.connect(dbname=args.database) as connection:
        cells = fetch_cells(connection)
        observations, problems = extract(cells)
        features = build_features(observations)
        observations.to_csv(args.output / "observations.csv", index=False)
        problems.to_csv(args.output / "quality_issues.csv", index=False)
        features.to_json(args.output / "features.json", orient="records", indent=2)
        summary = {
            "bronze_cells": len(cells),
            "measurements": len(observations),
            "valid_measurements": int(observations.quality_status.eq("valid").sum()),
            "quality_issues": len(problems),
            "province_years": len(observations[["province", "year"]].drop_duplicates()),
            "provinces": observations.province.nunique(),
            "feature_rows": len(features),
            "published": args.publish,
        }
        if args.publish:
            publish(connection, observations, features)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary))


if __name__ == "__main__":
    main()

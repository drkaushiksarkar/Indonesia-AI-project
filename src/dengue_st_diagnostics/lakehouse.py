from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from dengue_st_diagnostics.config import Settings


class Lakehouse:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.config = settings.section("lakehouse")
        environment_name = self.config["dsn_environment"]
        self.dsn = os.getenv(environment_name, f"dbname={self.config['database']}")

    def initialize(self) -> None:
        database = self.config["database"]
        admin = self.config["default_admin_database"]
        with psycopg.connect(f"dbname={admin}", autocommit=True) as connection:
            exists = connection.execute(
                "select 1 from pg_database where datname = %s",
                (database,),
            ).fetchone()
            if exists is None:
                connection.execute(sql.SQL("create database {}").format(sql.Identifier(database)))
        schema_path = Path(__file__).resolve().parents[2] / "sql" / "medallion.sql"
        with psycopg.connect(self.dsn, autocommit=True) as connection:
            connection.execute(schema_path.read_text(encoding="utf-8"))
            self._register_sources(connection)

    def _register_sources(self, connection: psycopg.Connection[Any]) -> None:
        rows = [
            (
                "existing_public_corpus",
                "Existing harvested public corpus",
                "Multiple public operators",
                "",
                "anonymous_public",
                "source_terms_apply",
                False,
                "available",
            ),
            (
                "local_curated_artifact",
                "Curated analytical artifact",
                "Local processing pipeline",
                "",
                "local_derived",
                "inherits_source_terms",
                False,
                "available",
            ),
            (
                "malaria_public_sismal",
                "Public SISMAL-derived malaria dashboard",
                "Ministry of Health Indonesia",
                "https://malaria.kemkes.go.id/case",
                "anonymous_public",
                "source_terms_apply",
                False,
                "available",
            ),
            (
                "malaria_public_library",
                "Public malaria document library",
                "Ministry of Health Indonesia",
                "https://malaria.kemkes.go.id/index.php/malaria-data",
                "anonymous_public",
                "source_terms_apply",
                False,
                "available",
            ),
            (
                "satusehat_public_tableau",
                "SATUSEHAT public aggregate dashboards",
                "Ministry of Health Indonesia",
                "https://satusehat.kemkes.go.id/data",
                "anonymous_public",
                "source_terms_apply",
                False,
                "available",
            ),
            (
                "skdr_public_live",
                "SKDR public live dashboard",
                "Ministry of Health Indonesia",
                "https://surkarkes.kemkes.go.id/dashboard-skdr-dev",
                "anonymous_public",
                "source_terms_apply",
                False,
                "available",
            ),
            (
                "silantor_public",
                "SILANTOR public vector dashboards",
                "Ministry of Health Indonesia",
                "https://silantor.kemkes.go.id/informasi-publik/dashboard",
                "anonymous_public",
                "source_terms_apply",
                False,
                "available_with_upstream_errors",
            ),
            (
                "satusehat_fhir",
                "SATUSEHAT FHIR",
                "Ministry of Health Indonesia",
                "https://api-satusehat.kemkes.go.id/fhir-r4/v1",
                "credentialed_api",
                "restricted",
                True,
                "credentials_required",
            ),
            (
                "sismal_internal_api",
                "SISMAL internal API",
                "Ministry of Health Indonesia",
                "https://sismal.kemkes.go.id/api",
                "credentialed_api",
                "restricted",
                True,
                "credentials_required",
            ),
        ]
        connection.cursor().executemany(
            """
            insert into metadata.source (
                source_id, source_name, operator_name, source_url, access_class,
                license_name, contains_person_level_data, credential_status
            ) values (%s, %s, %s, %s, %s, %s, %s, %s)
            on conflict (source_id) do update set
                source_name = excluded.source_name,
                operator_name = excluded.operator_name,
                source_url = excluded.source_url,
                access_class = excluded.access_class,
                license_name = excluded.license_name,
                contains_person_level_data = excluded.contains_person_level_data,
                credential_status = excluded.credential_status
            """,
            rows,
        )

    def load(self) -> None:
        csv.field_size_limit(sys.maxsize)
        with psycopg.connect(self.dsn) as connection:
            run_id = connection.execute(
                """
                insert into metadata.harvest_run (
                    started_at, pipeline_version, status, configuration, host_metadata
                ) values (%s, %s, %s, %s, %s) returning harvest_run_id
                """,
                (
                    datetime.now(UTC),
                    "1.0.0",
                    "running",
                    Jsonb(self.settings.values),
                    Jsonb(
                        {
                            "hostname": platform.node(),
                            "platform": platform.platform(),
                            "python": platform.python_version(),
                        }
                    ),
                ),
            ).fetchone()[0]
            self._load_blocked(connection)
            self._load_assets(connection, run_id)
            self._load_datasets(connection)
            self._transform(connection)
            self._quality(connection)
            connection.execute(
                """
                update metadata.harvest_run
                set completed_at = %s, status = 'completed'
                where harvest_run_id = %s
                """,
                (datetime.now(UTC), run_id),
            )
            connection.commit()

    def load_enrichment(self) -> None:
        csv.field_size_limit(sys.maxsize)
        with psycopg.connect(self.dsn) as connection:
            run_id = connection.execute(
                """
                insert into metadata.harvest_run (
                    started_at, pipeline_version, status, configuration, host_metadata
                ) values (%s, %s, %s, %s, %s) returning harvest_run_id
                """,
                (
                    datetime.now(UTC),
                    "1.1.0",
                    "running",
                    Jsonb(self.settings.values),
                    Jsonb(
                        {
                            "hostname": platform.node(),
                            "platform": platform.platform(),
                            "python": platform.python_version(),
                            "load_mode": "public_table_enrichment",
                        }
                    ),
                ),
            ).fetchone()[0]
            self._load_assets(connection, run_id)
            self._load_datasets(connection)
            self._transform(
                connection,
                selected={"public_enriched_observations"},
                replace=False,
            )
            self._quality(connection)
            connection.execute(
                """
                update metadata.harvest_run
                set completed_at = %s, status = 'completed'
                where harvest_run_id = %s
                """,
                (datetime.now(UTC), run_id),
            )
            connection.commit()

    def _load_blocked(self, connection: psycopg.Connection[Any]) -> None:
        path = self.settings.paths.quantitative / "credential_blocked_sources.csv"
        if not path.exists():
            return
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = [
                (row["source_id"], row["access_class"], row["reason"], row["endpoint"])
                for row in csv.DictReader(handle)
            ]
        connection.cursor().executemany(
            """
            insert into metadata.blocked_source (source_id, access_class, reason, endpoint)
            values (%s, %s, %s, %s)
            on conflict (source_id) do update set
                access_class = excluded.access_class,
                reason = excluded.reason,
                endpoint = excluded.endpoint,
                verified_at = now()
            """,
            rows,
        )

    def _asset_rows(
        self,
        run_id: int,
        registered_sizes: dict[str, set[int]],
        registered_relative_sizes: dict[str, set[int]],
    ) -> Iterator[tuple[Any, ...]]:
        dashboard_path = self.settings.paths.quantitative / "open_dashboard_assets.csv"
        known_paths: set[str] = set()
        if dashboard_path.exists():
            with dashboard_path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    absolute_path = row.get("path", "")
                    if absolute_path:
                        known_paths.add(str(Path(absolute_path).resolve()))
                    yield (
                        row["source_id"],
                        run_id,
                        row["dataset_name"],
                        row.get("request_url", ""),
                        row.get("request_method", "GET"),
                        Jsonb(json.loads(row.get("request_parameters") or "{}")),
                        self._timestamp(row.get("retrieved_at")),
                        self._integer(row.get("http_status")),
                        row.get("content_type", ""),
                        self._integer(row.get("byte_count")) or 0,
                        row.get("sha256", ""),
                        absolute_path,
                        row.get("status", ""),
                        row.get("error", ""),
                    )
        roots = [
            (self.settings.paths.raw, "existing_public_corpus"),
            (self.settings.paths.quantitative, "local_curated_artifact"),
        ]
        for root, source_id in roots:
            for path in sorted(root.rglob("*")):
                if not path.is_file() or str(path.resolve()) in known_paths:
                    continue
                resolved = str(path.resolve())
                size = path.stat().st_size
                relative = str(path.relative_to(self.settings.paths.root))
                if size in registered_sizes.get(resolved, set()) or size in (
                    registered_relative_sizes.get(relative, set())
                ):
                    continue
                digest = self._sha256(path)
                yield (
                    source_id,
                    run_id,
                    path.stem,
                    "",
                    "FILE",
                    Jsonb({}),
                    datetime.fromtimestamp(path.stat().st_mtime, UTC),
                    None,
                    "",
                    size,
                    digest,
                    resolved,
                    "cached",
                    "",
                )

    def _load_assets(self, connection: psycopg.Connection[Any], run_id: int) -> None:
        registered_sizes: defaultdict[str, set[int]] = defaultdict(set)
        registered_relative_sizes: defaultdict[str, set[int]] = defaultdict(set)
        for absolute_path, byte_count in connection.execute(
            "select absolute_path, byte_count from metadata.asset"
        ):
            registered_sizes[absolute_path].add(byte_count)
            relative = self._stored_relative_path(absolute_path)
            if relative:
                registered_relative_sizes[relative].add(byte_count)
        connection.cursor().executemany(
            """
            insert into metadata.asset (
                source_id, harvest_run_id, dataset_name, request_url, request_method,
                request_parameters, retrieved_at, http_status, content_type, byte_count,
                sha256, absolute_path, status, error_message
            ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            on conflict (source_id, request_url, absolute_path, sha256) do update set
                harvest_run_id = excluded.harvest_run_id,
                retrieved_at = excluded.retrieved_at,
                http_status = excluded.http_status,
                status = excluded.status,
                error_message = excluded.error_message
            """,
            self._asset_rows(
                run_id,
                dict(registered_sizes),
                dict(registered_relative_sizes),
            ),
        )

    def _load_datasets(self, connection: psycopg.Connection[Any]) -> None:
        registered_relative_sizes: defaultdict[str, set[int]] = defaultdict(set)
        for absolute_path, byte_count in connection.execute(
            """
            select d.absolute_path, a.byte_count
            from metadata.dataset d
            left join metadata.asset a using (asset_id)
            where d.medallion_layer = 'bronze'
            """
        ):
            relative = self._stored_relative_path(absolute_path)
            if relative and byte_count is not None:
                registered_relative_sizes[relative].add(byte_count)
        for path in sorted(self.settings.paths.quantitative.glob("*.csv")):
            relative = str(path.relative_to(self.settings.paths.root))
            if path.stat().st_size in registered_relative_sizes.get(relative, set()):
                continue
            self._load_dataset(connection, path)

    def _load_dataset(self, connection: psycopg.Connection[Any], path: Path) -> None:
        digest = self._sha256(path)
        absolute_path = str(path.resolve())
        asset = connection.execute(
            """
            select asset_id from metadata.asset
            where source_id = 'local_curated_artifact'
              and absolute_path = %s and sha256 = %s
            """,
            (absolute_path, digest),
        ).fetchone()
        asset_id = asset[0] if asset else None
        existing = connection.execute(
            """
            select dataset_id from metadata.dataset
            where medallion_layer = 'bronze' and absolute_path = %s and sha256 = %s
            """,
            (absolute_path, digest),
        ).fetchone()
        if existing is not None:
            connection.execute(
                "update metadata.dataset set asset_id = %s where dataset_id = %s",
                (asset_id, existing[0]),
            )
            return
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            dataset_id = connection.execute(
                """
                insert into metadata.dataset (
                    source_id, asset_id, dataset_name, medallion_layer, storage_format,
                    absolute_path, column_count, schema_json, sha256
                ) values (%s, %s, %s, 'bronze', 'csv', %s, %s, %s, %s)
                returning dataset_id
                """,
                (
                    "local_curated_artifact",
                    asset_id,
                    path.stem,
                    absolute_path,
                    len(columns),
                    Jsonb({"columns": columns}),
                    digest,
                ),
            ).fetchone()[0]
            row_count = 0
            with connection.cursor().copy(
                """
                copy bronze.record (dataset_id, row_number, source_row_hash, payload)
                from stdin
                """
            ) as copy:
                for row_count, row in enumerate(reader, start=1):
                    row = {
                        key: value.replace("\x00", "") if isinstance(value, str) else value
                        for key, value in row.items()
                        if key is not None
                    }
                    canonical = json.dumps(row, ensure_ascii=False, sort_keys=True)
                    row_hash = hashlib.sha256(canonical.encode()).hexdigest()
                    copy.write_row((dataset_id, row_count, row_hash, Jsonb(row)))
            connection.execute(
                "update metadata.dataset set row_count = %s where dataset_id = %s",
                (row_count, dataset_id),
            )

    def _transform(
        self,
        connection: psycopg.Connection[Any],
        selected: set[str] | None = None,
        replace: bool = True,
    ) -> None:
        if replace:
            connection.execute(
                "truncate metadata.lineage, gold.surveillance_fact, silver.observation restart identity"
            )
        selected = selected or {
            "canonical_cases",
            "malaria_puskesmas_indicators",
            "open_dashboard_records",
            "puskesmas_cases",
            "public_enriched_observations",
            "skdr_weekly_indicators",
            "vector_surveillance_indicators",
            "who_malaria_indicators",
        }
        for dataset_name in selected:
            rows = connection.execute(
                """
                with latest as (
                    select dataset_id
                    from metadata.dataset
                    where dataset_name = %s
                    order by loaded_at desc, dataset_id desc
                    limit 1
                )
                select b.bronze_record_id, b.payload, d.dataset_id, d.dataset_name
                from bronze.record b
                join metadata.dataset d using (dataset_id)
                join latest using (dataset_id)
                order by b.row_number
                """,
                (dataset_name,),
            )
            with connection.cursor().copy(
                """
                    copy silver.observation (
                        bronze_record_id, dataset_id, source_id, disease, metric,
                        value_numeric, value_text, unit, period_start, period_end,
                        temporal_resolution, admin_0, admin_1, admin_2, admin_3,
                        admin_4, spatial_resolution, facility_name, facility_type,
                        sex, age_group, species,
                    case_definition, confirmation_status, privacy_class,
                    provenance, record_hash
                ) from stdin
                """
            ) as copy:
                for bronze_id, payload, dataset_id, name in rows:
                    for observation in self._observations(name, payload):
                        upstream_bronze_id = observation.pop("_bronze_record_id", bronze_id)
                        supplied_record_hash = observation.pop("_record_hash", None)
                        provenance = {
                            "bronze_record_id": upstream_bronze_id,
                            "normalization_record_bronze_id": bronze_id,
                            "dataset_id": dataset_id,
                            "source_record": payload,
                            "transformation": "medallion_normalize_v2",
                        }
                        values = (
                            upstream_bronze_id,
                            dataset_id,
                            observation["source_id"],
                            observation.get("disease"),
                            observation["metric"],
                            observation.get("value_numeric"),
                            observation.get("value_text"),
                            observation.get("unit"),
                            observation.get("period_start"),
                            observation.get("period_end"),
                            observation.get("temporal_resolution"),
                            observation.get("admin_0"),
                            observation.get("admin_1"),
                            observation.get("admin_2"),
                            observation.get("admin_3"),
                            observation.get("admin_4"),
                            observation.get("spatial_resolution"),
                            observation.get("facility_name"),
                            observation.get("facility_type"),
                            observation.get("sex"),
                            observation.get("age_group"),
                            observation.get("species"),
                            observation.get("case_definition"),
                            observation.get("confirmation_status"),
                            observation.get("privacy_class", "aggregate"),
                            Jsonb(provenance),
                            supplied_record_hash
                            or self._record_hash(upstream_bronze_id, observation),
                        )
                        copy.write_row(values)
        connection.execute(
            """
            insert into gold.surveillance_fact (
                silver_observation_id, disease, metric, value, unit, period_start,
                period_end, temporal_resolution, admin_0, admin_1, admin_2, admin_3,
                admin_4, spatial_resolution, facility_name, facility_type, sex, age_group,
                species, case_definition,
                confirmation_status, source_id, source_dataset_id, provenance, fact_hash
            )
            select
                silver_observation_id, disease, metric, value_numeric, unit, period_start,
                period_end, temporal_resolution, admin_0, admin_1, admin_2, admin_3,
                admin_4, spatial_resolution, facility_name, facility_type, sex, age_group,
                species, case_definition,
                confirmation_status, source_id, dataset_id, provenance, record_hash
            from silver.observation
            where value_numeric is not null
            on conflict (fact_hash) do nothing
            """
        )
        connection.execute(
            """
            insert into metadata.lineage (
                bronze_record_id, silver_observation_id, gold_fact_id,
                transformation_name, transformation_version
            )
            select
                s.bronze_record_id, s.silver_observation_id, g.gold_fact_id,
                'medallion_normalize', '1.0.0'
            from silver.observation s
            left join gold.surveillance_fact g
                on g.silver_observation_id = s.silver_observation_id
            where not exists (
                select 1 from metadata.lineage l
                where l.silver_observation_id = s.silver_observation_id
            )
            """
        )

    def _observations(self, dataset_name: str, row: dict[str, Any]) -> Iterable[dict[str, Any]]:
        if dataset_name == "canonical_cases":
            yield self._base_observation(
                row,
                source_id="local_curated_artifact",
                disease="dengue",
                metric="cases",
                value=row.get("cases"),
                admin_0=row.get("admin_0"),
                admin_1=row.get("admin_1"),
                admin_2=row.get("admin_2"),
                admin_3=row.get("admin_3"),
            )
        elif dataset_name == "puskesmas_cases":
            for metric in ["cases", "deaths", "male_cases", "female_cases"]:
                if self._numeric(row.get(metric)) is not None:
                    yield self._base_observation(
                        row,
                        source_id="local_curated_artifact",
                        disease="dengue",
                        metric=metric,
                        value=row.get(metric),
                        admin_0="Indonesia",
                        admin_1=row.get("province"),
                        admin_2=row.get("district_city"),
                        admin_3=row.get("kecamatan"),
                        facility_name=row.get("puskesmas"),
                        facility_type="puskesmas",
                    )
        elif dataset_name == "malaria_puskesmas_indicators":
            yield self._base_observation(
                row,
                source_id="local_curated_artifact",
                disease="malaria",
                metric=row.get("metric") or "malaria_indicator",
                value=row.get("value"),
                admin_0="Indonesia",
                facility_name=row.get("puskesmas"),
                facility_type="puskesmas",
            )
        elif dataset_name == "vector_surveillance_indicators":
            fields = [
                "aedes_houses_inspected",
                "aedes_houses_positive",
                "aedes_house_index_free_percent",
                "anopheles_habitats_inspected",
                "anopheles_habitats_positive",
                "anopheles_habitat_index",
            ]
            for metric in fields:
                if self._numeric(row.get(metric)) is not None:
                    yield self._base_observation(
                        row,
                        source_id="local_curated_artifact",
                        disease="dengue" if metric.startswith("aedes") else "malaria",
                        metric=metric,
                        value=row.get(metric),
                        admin_0="Indonesia",
                        admin_3=row.get("locality"),
                        facility_name=row.get("puskesmas"),
                        facility_type="puskesmas",
                    )
        elif dataset_name == "skdr_weekly_indicators":
            year = self._integer(row.get("report_year"))
            week = self._integer(row.get("report_week"))
            start = date.fromisocalendar(year, week, 1) if year and week else None
            indicator = row.get("indicator") or "reported_cases"
            yield self._base_observation(
                row,
                source_id="local_curated_artifact",
                disease="dengue"
                if "dengue" in indicator.casefold()
                else "malaria"
                if "malaria" in indicator.casefold()
                else None,
                metric=indicator,
                value=row.get("cumulative_reports"),
                admin_0="Indonesia",
                period_start=start,
                period_end=start + timedelta(days=6) if start else None,
                temporal_resolution="week_cumulative_snapshot",
            )
        elif dataset_name == "who_malaria_indicators":
            yield self._base_observation(
                row,
                source_id="local_curated_artifact",
                disease="malaria",
                metric=row.get("indicator_name") or row.get("indicator_code") or "who_indicator",
                value=row.get("numeric_value") or row.get("value"),
                admin_0="Indonesia",
                period_start=f"{row.get('year')}-01-01" if row.get("year") else None,
                period_end=f"{row.get('year')}-12-31" if row.get("year") else None,
                temporal_resolution="year",
            )
        elif dataset_name == "open_dashboard_records":
            yield from self._dashboard_observations(row)
        elif dataset_name == "public_enriched_observations":
            yield {
                "source_id": row.get("source_id") or "existing_public_corpus",
                "disease": row.get("disease"),
                "metric": row.get("metric") or "source_measurement",
                "value_numeric": self._numeric(row.get("value_numeric")),
                "value_text": row.get("value_text"),
                "unit": row.get("unit"),
                "period_start": self._date(row.get("period_start")),
                "period_end": self._date(row.get("period_end")),
                "temporal_resolution": row.get("temporal_resolution") or None,
                "admin_0": row.get("admin_0") or "Indonesia",
                "admin_1": row.get("admin_1") or None,
                "admin_2": row.get("admin_2") or None,
                "admin_3": row.get("admin_3") or None,
                "admin_4": row.get("admin_4") or None,
                "spatial_resolution": row.get("spatial_resolution") or None,
                "facility_name": row.get("facility_name") or None,
                "facility_type": row.get("facility_type") or None,
                "sex": row.get("sex") or None,
                "age_group": row.get("age_group") or None,
                "species": row.get("species") or None,
                "case_definition": row.get("case_definition") or None,
                "confirmation_status": row.get("confirmation_status") or None,
                "privacy_class": row.get("privacy_class") or "aggregate",
                "_bronze_record_id": self._integer(row.get("upstream_bronze_record_id")),
                "_record_hash": row.get("observation_hash"),
            }

    def _dashboard_observations(self, row: dict[str, Any]) -> Iterable[dict[str, Any]]:
        try:
            payload = json.loads(row.get("record_json") or "{}")
        except json.JSONDecodeError:
            return
        dataset_name = row.get("dataset_name", "")
        source_id = row.get("source_id") or "local_curated_artifact"
        retrieved_at = self._timestamp(row.get("retrieved_at"))
        retrieved_date = retrieved_at.date() if retrieved_at else datetime.now(UTC).date()
        if source_id == "skdr_public_live":
            yield from self._skdr_dashboard_observations(source_id, payload, retrieved_date)
            return
        year = self._integer(payload.get("thn_lap")) or self._year_from_name(dataset_name)
        period_start = f"{year}-01-01" if year else None
        is_year_to_date = bool(year and year >= retrieved_date.year)
        period_end = (
            retrieved_date.isoformat() if is_year_to_date else f"{year}-12-31" if year else None
        )
        excluded = {
            "data",
            "endemisitas",
            "id_kabupaten",
            "idprop1",
            "last_update",
            "nama_kabupaten",
            "namaprop1",
            "thn_lap",
        }
        for metric, value in payload.items():
            numeric = self._numeric(value)
            if metric in excluded or numeric is None:
                continue
            yield {
                "source_id": source_id,
                "disease": "malaria" if source_id == "malaria_public_sismal" else None,
                "metric": metric,
                "value_numeric": numeric,
                "unit": None,
                "period_start": self._date(period_start),
                "period_end": self._date(period_end),
                "temporal_resolution": "year_to_date"
                if is_year_to_date
                else "year"
                if year
                else "dashboard_snapshot",
                "admin_0": "Indonesia",
                "admin_1": payload.get("namaprop1"),
                "admin_2": payload.get("nama_kabupaten"),
                "confirmation_status": (
                    "SISMAL reported" if source_id == "malaria_public_sismal" else None
                ),
                "privacy_class": "aggregate",
            }

    def _skdr_dashboard_observations(
        self,
        source_id: str,
        payload: dict[str, Any],
        retrieved_date: date,
    ) -> Iterable[dict[str, Any]]:
        year = self._integer(payload.get("year"))
        week = self._integer(payload.get("week"))
        start = date.fromisocalendar(year, week, 1) if year and week else None
        period_start = start or (date(year, 1, 1) if year else None)
        period_end = start + timedelta(days=6) if start else date(year, 12, 31) if year else None
        if period_end and period_end > retrieved_date:
            period_end = retrieved_date
        disease_code = str(payload.get("disease_code") or "")
        disease = "dengue" if disease_code == "16SDF" else "malaria"
        for metric in ["cases", "outbreak_cases"]:
            numeric = self._numeric(payload.get(metric))
            if numeric is None:
                continue
            yield {
                "source_id": source_id,
                "disease": disease,
                "metric": metric,
                "value_numeric": numeric,
                "period_start": period_start,
                "period_end": period_end,
                "temporal_resolution": payload.get("temporal_resolution"),
                "admin_0": "Indonesia",
                "admin_1": payload.get("province"),
                "case_definition": "suspected" if disease == "dengue" else "confirmed",
                "confirmation_status": "SKDR reported",
                "privacy_class": "aggregate",
            }

    def _base_observation(
        self,
        row: dict[str, Any],
        source_id: str,
        disease: str | None,
        metric: str,
        value: Any,
        admin_0: Any = None,
        admin_1: Any = None,
        admin_2: Any = None,
        admin_3: Any = None,
        facility_name: Any = None,
        facility_type: Any = None,
        period_start: Any = None,
        period_end: Any = None,
        temporal_resolution: Any = None,
    ) -> dict[str, Any]:
        return {
            "source_id": source_id,
            "disease": disease,
            "metric": metric,
            "value_numeric": self._numeric(value),
            "unit": row.get("unit") or row.get("satuan"),
            "period_start": self._date(period_start or row.get("period_start")),
            "period_end": self._date(period_end or row.get("period_end")),
            "temporal_resolution": temporal_resolution or row.get("temporal_resolution"),
            "admin_0": admin_0,
            "admin_1": admin_1,
            "admin_2": admin_2,
            "admin_3": admin_3,
            "admin_4": row.get("admin_4"),
            "spatial_resolution": row.get("spatial_resolution"),
            "facility_name": facility_name,
            "facility_type": facility_type,
            "case_definition": row.get("case_definition"),
            "confirmation_status": row.get("confirmation_status"),
            "privacy_class": row.get("privacy_class") or "aggregate",
        }

    def _quality(self, connection: psycopg.Connection[Any]) -> None:
        connection.execute("delete from metadata.quality_result")
        checks = [
            (
                "bronze",
                "records_have_source_hash",
                "critical",
                "select count(*) from bronze.record where source_row_hash = ''",
                0,
            ),
            (
                "silver",
                "observations_have_metric",
                "critical",
                "select count(*) from silver.observation where metric = ''",
                0,
            ),
            (
                "silver",
                "observations_have_provenance",
                "critical",
                "select count(*) from silver.observation where provenance = '{}'::jsonb",
                0,
            ),
            (
                "gold",
                "facts_have_silver_parent",
                "critical",
                """
                select count(*) from gold.surveillance_fact g
                left join silver.observation s using (silver_observation_id)
                where s.silver_observation_id is null
                """,
                0,
            ),
            (
                "gold",
                "nonnegative_surveillance_values",
                "high",
                """
                select count(*) from gold.surveillance_fact
                where value < 0
                  and lower(metric) not like '%percent%'
                  and lower(metric) not like '%change%'
                """,
                0,
            ),
        ]
        rows = []
        for layer, name, severity, query, expected in checks:
            observed = connection.execute(query).fetchone()[0]
            rows.append(
                (
                    layer,
                    name,
                    severity,
                    observed == expected,
                    float(observed),
                    str(expected),
                    Jsonb({}),
                )
            )
        connection.cursor().executemany(
            """
            insert into metadata.quality_result (
                layer_name, check_name, severity, passed, observed_value,
                expected_value, details
            ) values (%s, %s, %s, %s, %s, %s, %s)
            """,
            rows,
        )

    def export(self) -> Path:
        root = Path(self.config["export_directory"]).resolve()
        root.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["pg_dump", "--format=custom", "--file", str(root / "lakehouse.dump"), self.dsn],
            check=True,
        )
        subprocess.run(
            [
                "pg_dump",
                "--schema-only",
                "--format=plain",
                "--file",
                str(root / "schema.sql"),
                self.dsn,
            ],
            check=True,
        )
        with psycopg.connect(self.dsn) as connection:
            tables = connection.execute(
                """
                select table_schema, table_name
                from information_schema.tables
                where table_schema in ('metadata', 'bronze', 'silver', 'gold')
                  and table_type = 'BASE TABLE'
                order by table_schema, table_name
                """
            ).fetchall()
            for schema_name, table_name in tables:
                path = root / "csv" / schema_name / f"{table_name}.csv.gz"
                path.parent.mkdir(parents=True, exist_ok=True)
                query = sql.SQL("copy {}.{} to stdout with (format csv, header true)").format(
                    sql.Identifier(schema_name),
                    sql.Identifier(table_name),
                )
                with gzip.open(path, "wb") as handle, connection.cursor().copy(query) as copy:
                    for block in copy:
                        handle.write(bytes(block))
            for relation in [
                "silver.observation",
                "gold.surveillance_fact",
                "gold.native_observation",
                "gold.modeling_series",
                "gold.aggregated_series",
                "gold.grain_inventory",
                "gold.source_coverage",
                "gold.diagnostic_statistic",
                "gold.diagnostic_component",
                "gold.diagnostic_event",
                "metadata.diagnostic_run",
                "metadata.diagnostic_method",
                "metadata.diagnostic_series",
                "metadata.diagnostic_input",
                "metadata.diagnostic_artifact",
                "metadata.asset_provenance",
            ]:
                cursor = connection.execute(f"select * from {relation}")
                columns = [column.name for column in cursor.description or []]
                frame = pd.DataFrame.from_records(cursor.fetchall(), columns=columns)
                for column in frame.select_dtypes(include=["object", "str"]).columns:
                    contains_nested = (
                        frame[column]
                        .map(lambda value: isinstance(value, (dict, list, tuple)))
                        .any()
                    )
                    if contains_nested:
                        frame[column] = frame[column].map(
                            lambda value: (
                                json.dumps(
                                    value,
                                    default=str,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                )
                                if isinstance(value, (dict, list, tuple))
                                else value
                            )
                        )
                path = root / "parquet" / f"{relation.replace('.', '__')}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(path, index=False)
        manifest = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                manifest.append(
                    {
                        "path": str(path.relative_to(root)),
                        "bytes": path.stat().st_size,
                        "sha256": self._sha256(path),
                    }
                )
        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest_path

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _stored_relative_path(value: str) -> str:
        normalized = value.replace("\\", "/")
        marker = "/outputs/"
        if marker not in normalized:
            return ""
        return normalized.split(marker, 1)[1]

    @staticmethod
    def _numeric(value: Any) -> float | None:
        if value is None or value == "":
            return None
        text = str(value).strip().replace(" ", "")
        if text.count(",") == 1 and "." not in text:
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _integer(value: Any) -> int | None:
        numeric = Lakehouse._numeric(value)
        return int(numeric) if numeric is not None else None

    @staticmethod
    def _date(value: Any) -> date | None:
        if isinstance(value, date):
            return value
        if value is None or value == "":
            return None
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            return None

    @staticmethod
    def _timestamp(value: Any) -> datetime | None:
        if value is None or value == "":
            return None
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return None

    @staticmethod
    def _year_from_name(value: str) -> int | None:
        match = re.search(r"(?:19|20)\d{2}", value)
        return int(match.group()) if match else None

    @staticmethod
    def _record_hash(bronze_id: int, observation: dict[str, Any]) -> str:
        value = json.dumps(
            {"bronze_record_id": bronze_id, "observation": observation},
            default=str,
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(value.encode()).hexdigest()

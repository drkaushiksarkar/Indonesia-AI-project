from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests

from dengue_st_diagnostics.artifacts import (
    create_figures,
    create_public_harvest_figure,
    create_puskesmas_figures,
    create_surveillance_figures,
)
from dengue_st_diagnostics.config import Settings
from dengue_st_diagnostics.diagnostics import (
    diagnose_quality,
    diagnose_spatial,
    diagnose_spatiotemporal,
    diagnose_temporal,
    method_registry,
)
from dengue_st_diagnostics.harvest import (
    harvest_boundaries,
    harvest_ckan,
    harvest_gbif,
    harvest_kemkes,
    harvest_ncbi,
    harvest_opendengue,
    harvest_portals,
    harvest_public_sources,
    harvest_puskesmas,
    harvest_satusehat,
    harvest_who,
)
from dengue_st_diagnostics.harvest import (
    harvest_literature as harvest_literature_records,
)
from dengue_st_diagnostics.harvest.portal_normalize import normalize_portal_records
from dengue_st_diagnostics.harvest.surveillance import parse_skdr, source_catalog
from dengue_st_diagnostics.io import sha256, write_frame, write_json
from dengue_st_diagnostics.multiscale import build_multiscale, grain_inventory
from dengue_st_diagnostics.normalize import normalize_paths
from dengue_st_diagnostics.schema import combine
from dengue_st_diagnostics.surveillance_quality import surveillance_quality


def _save_table(frame: pd.DataFrame, root: Path, name: str) -> list[Path]:
    if frame.empty and len(frame.columns) == 0:
        frame = pd.DataFrame({"status": pd.Series(dtype="string")})
    return [
        write_frame(frame, root / f"{name}.csv"),
        write_frame(frame, root / f"{name}.parquet"),
    ]


def _reconciliation(cases: pd.DataFrame) -> pd.DataFrame:
    if cases.empty:
        return pd.DataFrame()
    frame = cases.loc[cases["period_start"].notna() & cases["cases"].notna()].copy()
    frame["year"] = frame["period_start"].dt.year
    totals = (
        frame.groupby(
            ["source", "admin_level", "temporal_resolution", "year"],
            dropna=False,
            as_index=False,
        )["cases"]
        .sum()
        .rename(columns={"cases": "total_cases"})
    )
    rows: list[dict[str, Any]] = []
    for keys, group in totals.groupby(["admin_level", "temporal_resolution", "year"], dropna=False):
        for left_index, left in enumerate(group.itertuples(index=False)):
            for right in list(group.itertuples(index=False))[left_index + 1 :]:
                denominator = max(float(left.total_cases), float(right.total_cases), 1.0)
                rows.append(
                    {
                        "admin_level": keys[0],
                        "temporal_resolution": keys[1],
                        "year": keys[2],
                        "source_a": left.source,
                        "source_b": right.source,
                        "total_cases_a": left.total_cases,
                        "total_cases_b": right.total_cases,
                        "absolute_difference": abs(left.total_cases - right.total_cases),
                        "relative_difference": abs(left.total_cases - right.total_cases)
                        / denominator,
                    }
                )
    return pd.DataFrame.from_records(rows)


def _tag_literature(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    value = frame.copy()
    text = (value["title"].fillna("") + " " + value["abstract"].fillna("")).str.casefold()
    tags = {
        "bayesian": r"bayes|conditional autoregressive|\bcar\b|inla",
        "scan_statistic": r"scan statistic|satscan|flexscan",
        "spatial_autocorrelation": r"moran|geary|getis|lisa|autocorrelation",
        "wavelet": r"wavelet|periodicity|phase angle",
        "clustering": r"dynamic time warping|dtw|cluster",
        "lag_response": r"distributed lag|dlnm|lagged",
        "machine_learning": r"deep learning|machine learning|neural network|transfer learning",
        "disaggregation": r"disaggregat|downscal|small area",
        "varying_coefficient": r"varying coefficient|gamm|generalized additive",
        "hotspot": r"hot ?spot|emerging hot",
    }
    value["method_tags"] = [
        "; ".join(
            name
            for name, pattern in tags.items()
            if pd.Series([document]).str.contains(pattern, regex=True).iloc[0]
        )
        for document in text
    ]
    value["dengue_relevant"] = text.str.contains(r"dengue|demam berdarah", regex=True)
    value["indonesia_relevant"] = text.str.contains(
        r"indonesia|jakarta|java|yogyakarta|bandung|makassar", regex=True
    )
    return value


def _execution_status(
    registry: pd.DataFrame,
    temporal: pd.DataFrame,
    global_spatial: pd.DataFrame,
    local_spatial: pd.DataFrame,
    spatiotemporal: dict[str, pd.DataFrame],
    reconciliation: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    computed = {
        "grain_inventory": not inventory.empty,
        "completeness_profile": not inventory.empty,
        "cross_source_reconciliation": not reconciliation.empty,
        "zero_overdispersion": not temporal.empty,
        "adf_kpss": not temporal.empty,
        "acf_ljung_box": not temporal.empty,
        "stl": not temporal.empty,
        "periodogram": not temporal.empty,
        "continuous_wavelet": not temporal.empty,
        "pelt_change_points": not temporal.empty,
        "global_moran": not global_spatial.empty,
        "geary_c": not global_spatial.empty,
        "local_moran_lisa": not local_spatial.empty,
        "getis_ord_gi_star": not local_spatial.empty,
        "semivariogram": not global_spatial.empty,
        "spatial_scan_statistic": (
            not global_spatial.empty
            and global_spatial["method"].eq("space_time_scan_statistic").any()
        ),
        "emerging_hotspot": not spatiotemporal["emerging_hotspots"].empty,
        "dtw_clustering": not spatiotemporal["dtw_clusters"].empty,
        "seasonal_phase": not spatiotemporal["seasonal_phase"].empty,
        "space_time_variance": not spatiotemporal["variance_decomposition"].empty,
        "domain_similarity": not spatiotemporal["transfer_similarity"].empty,
        "maup_sensitivity": inventory["spatial_resolution"].nunique() > 1
        if not inventory.empty
        else False,
        "temporal_aggregation_sensitivity": inventory["temporal_resolution"].nunique() > 1
        if not inventory.empty
        else False,
        "population_offset_audit": not inventory.empty,
    }
    value = registry.copy()
    value["status"] = value["method"].map(
        lambda method: "computed" if computed.get(method, False) else "missing_input"
    )
    return value


def run_pipeline(
    settings: Settings,
    input_paths: list[Path] | None = None,
    download_sources: bool = True,
    harvest_literature: bool = True,
) -> dict[str, Path]:
    session = requests.Session()
    session.headers.update({"User-Agent": "dengue-st-diagnostics/0.1"})
    opendengue, opendengue_sources = harvest_opendengue(settings, session, download_sources)
    boundaries, boundary_sources = harvest_boundaries(settings, session, download_sources)
    ckan_files, ckan_searches, ckan_resources = harvest_ckan(settings, session, download_sources)
    kemkes_documents, kemkes_sources = harvest_kemkes(settings, session, download_sources)
    who, who_status = harvest_who(settings, session, download_sources)
    sequences, sequence_status = harvest_ncbi(settings, session, download_sources)
    vectors, vector_summary, vector_status = harvest_gbif(settings, session, download_sources)
    portal_resources, portal_tables, portal_cells, portal_searches = harvest_portals(
        settings, session, download_sources
    )
    public = harvest_public_sources(settings, download_sources)
    malaria_species, malaria_puskesmas, vector_surveillance = normalize_portal_records(
        portal_resources
    )
    satusehat, satusehat_status = harvest_satusehat(settings, session, download_sources)
    skdr_weekly, skdr_ebs = parse_skdr(kemkes_documents)
    surveillance_quality_frame = surveillance_quality(
        who,
        sequences,
        vectors,
        portal_resources,
        skdr_weekly,
        malaria_species,
    )
    puskesmas_cases, puskesmas_records, puskesmas_catalog, puskesmas_readiness = harvest_puskesmas(
        settings, session, download_sources
    )
    literature, literature_searches = harvest_literature_records(
        settings,
        session,
        harvest_literature,
    )
    paths = [*ckan_files, *(input_paths or [])]
    local_cases, normalization = normalize_paths(paths)
    cases = combine([opendengue, local_cases, puskesmas_cases])
    analysis = settings.section("analysis")
    multiscale = build_multiscale(cases, analysis["temporal_grains"])
    inventory = grain_inventory(multiscale)
    quality, coverage = diagnose_quality(cases, multiscale)
    temporal = diagnose_temporal(multiscale, int(analysis["minimum_periods"]))
    global_spatial, local_spatial, spatial_matches = diagnose_spatial(
        multiscale,
        boundaries,
        int(analysis["monte_carlo_permutations"]),
        int(analysis["random_seed"]),
    )
    spatiotemporal = diagnose_spatiotemporal(multiscale, local_spatial)
    reconciliation = _reconciliation(cases)
    literature = _tag_literature(literature)
    registry = method_registry()
    execution = _execution_status(
        registry,
        temporal,
        global_spatial,
        local_spatial,
        spatiotemporal,
        reconciliation,
        inventory,
    )
    manifest_columns = ["source", "url", "path", "status", "sha256", "error"]
    puskesmas_sources = (
        puskesmas_catalog.rename(columns={"dataset_title": "source"})[manifest_columns]
        if not puskesmas_catalog.empty
        else pd.DataFrame(columns=manifest_columns)
    )
    source_manifest = pd.concat(
        [
            opendengue_sources,
            boundary_sources,
            kemkes_sources,
            puskesmas_sources,
            public["public_source_resources"][manifest_columns]
            if not public["public_source_resources"].empty
            else pd.DataFrame(columns=manifest_columns),
        ],
        ignore_index=True,
    )
    tables = {
        "canonical_cases": cases,
        "multiscale_cases": multiscale,
        "grain_inventory": inventory,
        "quality_checks": quality,
        "temporal_coverage": coverage,
        "temporal_diagnostics": temporal,
        "global_spatial_diagnostics": global_spatial,
        "local_spatial_diagnostics": local_spatial,
        "spatial_boundary_matches": spatial_matches,
        "cross_source_reconciliation": reconciliation,
        "method_registry": registry,
        "method_execution_status": execution,
        "literature_2020_2025": literature,
        "literature_search_status": literature_searches,
        "ckan_search_status": ckan_searches,
        "ckan_resources": ckan_resources,
        "source_manifest": source_manifest,
        "normalization_status": normalization,
        "puskesmas_cases": puskesmas_records,
        "puskesmas_source_catalog": puskesmas_catalog,
        "puskesmas_modeling_readiness": puskesmas_readiness,
        "surveillance_source_catalog": source_catalog(),
        "who_malaria_indicators": who,
        "who_api_status": who_status,
        "pathogen_sequence_metadata": sequences,
        "pathogen_sequence_api_status": sequence_status,
        "vector_occurrences": vectors,
        "vector_occurrence_summary": vector_summary,
        "vector_api_status": vector_status,
        "surveillance_portal_resources": portal_resources,
        "surveillance_portal_tables": portal_tables,
        "surveillance_portal_cells": portal_cells,
        "surveillance_portal_search_status": portal_searches,
        "malaria_species_surveillance": malaria_species,
        "malaria_puskesmas_indicators": malaria_puskesmas,
        "vector_surveillance_indicators": vector_surveillance,
        "satusehat_aggregate_conditions": satusehat,
        "satusehat_api_status": satusehat_status,
        "skdr_weekly_indicators": skdr_weekly,
        "skdr_ebs_indicators": skdr_ebs,
        "surveillance_data_quality": surveillance_quality_frame,
    }
    tables.update(spatiotemporal)
    tables.update(public)
    if not kemkes_documents.empty:
        tables["kemkes_document_catalog"] = kemkes_documents.drop(columns=["text"])
        tables["kemkes_document_text"] = kemkes_documents
    created: list[Path] = []
    for name, frame in tables.items():
        created.extend(_save_table(frame, settings.paths.quantitative, name))
    images = create_figures(
        settings.paths.images,
        inventory,
        multiscale,
        global_spatial,
        spatiotemporal,
        boundaries,
    )
    images.extend(create_puskesmas_figures(settings.paths.images, puskesmas_records))
    images.extend(create_surveillance_figures(settings.paths.images, vectors, sequences, who))
    images.extend(
        create_public_harvest_figure(
            settings.paths.images,
            public["public_source_resources"],
            public["public_dataset_catalog"],
            public["public_dashboard_status"],
        )
    )
    created.extend(images)
    manifest = pd.DataFrame(
        [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "artifact_type": "image" if path.suffix == ".png" else "quantitative",
            }
            for path in created
        ]
    )
    created.extend(_save_table(manifest, settings.paths.root, "artifact_manifest"))
    summary = {
        "case_rows": len(cases),
        "puskesmas_case_rows": int(puskesmas_records["aggregation_level"].eq("puskesmas").sum())
        if not puskesmas_records.empty
        else 0,
        "puskesmas_facilities": int(puskesmas_records["puskesmas"].nunique())
        if not puskesmas_records.empty
        else 0,
        "panels": int(multiscale["panel_id"].nunique()) if not multiscale.empty else 0,
        "literature_records": len(literature),
        "who_malaria_records": len(who),
        "pathogen_sequence_records": len(sequences),
        "vector_occurrence_records": len(vectors),
        "surveillance_portal_resources": len(portal_resources),
        "surveillance_portal_cells": len(portal_cells),
        "malaria_species_records": len(malaria_species),
        "malaria_puskesmas_records": len(malaria_puskesmas),
        "vector_surveillance_records": len(vector_surveillance),
        "skdr_weekly_indicator_records": len(skdr_weekly),
        "skdr_ebs_records": len(skdr_ebs),
        "public_datasets": len(public["public_dataset_catalog"]),
        "public_repository_records": len(public["public_repository_catalog"]),
        "public_resources": len(public["public_source_resources"]),
        "public_downloaded_resources": int(
            public["public_source_resources"]["status"].isin(["cached", "downloaded"]).sum()
        )
        if not public["public_source_resources"].empty
        else 0,
        "public_tabular_cells": len(public["public_tabular_cells"]),
        "public_document_pages": len(public["public_document_pages"]),
        "public_dashboard_records": len(public["public_dashboard_records"]),
        "quantitative_artifacts": int(sum(path.suffix in {".csv", ".parquet"} for path in created)),
        "image_artifacts": len(images),
    }
    summary_path = write_json(summary, settings.paths.root / "run_summary.json")
    return {
        "root": settings.paths.root,
        "manifest": settings.paths.root / "artifact_manifest.csv",
        "summary": summary_path,
    }

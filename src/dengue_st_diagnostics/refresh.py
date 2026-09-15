from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import requests

from dengue_st_diagnostics.config import Settings
from dengue_st_diagnostics.harvest.kemkes import harvest_kemkes
from dengue_st_diagnostics.harvest.portal_normalize import normalize_portal_records
from dengue_st_diagnostics.harvest.public_dashboards import harvest_open_dashboards
from dengue_st_diagnostics.harvest.public_sources import (
    harvest_public_repositories,
    harvest_public_sources,
)
from dengue_st_diagnostics.harvest.surveillance import parse_skdr
from dengue_st_diagnostics.surveillance_quality import surveillance_quality


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return path


def _write_json(value: object, path: Path) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_table(frame: pd.DataFrame, root: Path, name: str) -> list[Path]:
    if frame.empty and len(frame.columns) == 0:
        frame = pd.DataFrame({"status": pd.Series(dtype="string")})
    return [
        _write_frame(frame, root / f"{name}.csv"),
        _write_frame(frame, root / f"{name}.parquet"),
    ]


def _read(root: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(root / f"{name}.parquet")


def refresh_surveillance_artifacts(settings: Settings) -> dict[str, Path]:
    root = settings.paths.root
    quantitative = settings.paths.quantitative
    kemkes = _read(quantitative, "kemkes_document_text")
    portal = _read(quantitative, "surveillance_portal_resources")
    who = _read(quantitative, "who_malaria_indicators")
    sequences = _read(quantitative, "pathogen_sequence_metadata")
    vectors = _read(quantitative, "vector_occurrences")
    skdr_weekly, skdr_ebs = parse_skdr(kemkes)
    malaria_species, malaria_puskesmas, vector_surveillance = normalize_portal_records(portal)
    quality = surveillance_quality(
        who,
        sequences,
        vectors,
        portal,
        skdr_weekly,
        malaria_species,
    )
    frames = {
        "skdr_weekly_indicators": skdr_weekly,
        "skdr_ebs_indicators": skdr_ebs,
        "malaria_species_surveillance": malaria_species,
        "malaria_puskesmas_indicators": malaria_puskesmas,
        "vector_surveillance_indicators": vector_surveillance,
        "surveillance_data_quality": quality,
    }
    for name, frame in frames.items():
        _save_table(frame, quantitative, name)
    summary_path = root / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "malaria_species_records": len(malaria_species),
            "malaria_puskesmas_records": len(malaria_puskesmas),
            "vector_surveillance_records": len(vector_surveillance),
            "skdr_weekly_indicator_records": len(skdr_weekly),
            "skdr_ebs_records": len(skdr_ebs),
        }
    )
    summary["quantitative_artifacts"] = len(list(quantitative.glob("*.csv"))) + len(
        list(quantitative.glob("*.parquet"))
    )
    summary["image_artifacts"] = len(list(settings.paths.images.glob("*.png")))
    _write_json(summary, summary_path)
    artifacts = sorted(
        [
            *quantitative.glob("*.csv"),
            *quantitative.glob("*.parquet"),
            *settings.paths.images.glob("*.png"),
        ]
    )
    manifest = pd.DataFrame.from_records(
        [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "artifact_type": "image" if path.suffix == ".png" else "quantitative",
            }
            for path in artifacts
        ]
    )
    manifest_csv = _write_frame(manifest, root / "artifact_manifest.csv")
    _write_frame(manifest, root / "artifact_manifest.parquet")
    return {"root": root, "manifest": manifest_csv, "summary": summary_path}


def _finalize_public_artifacts(
    settings: Settings,
    frames: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    root = settings.paths.root
    quantitative = settings.paths.quantitative
    from dengue_st_diagnostics.artifacts.public_figures import (
        create_public_harvest_figure,
    )

    create_public_harvest_figure(
        settings.paths.images,
        frames["public_source_resources"],
        frames["public_dataset_catalog"],
        frames["public_dashboard_status"],
    )
    summary_path = root / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    resources = frames["public_source_resources"]
    summary.update(
        {
            "public_datasets": len(frames["public_dataset_catalog"]),
            "public_repository_records": len(frames["public_repository_catalog"]),
            "public_resources": len(resources),
            "public_downloaded_resources": int(
                resources["status"].isin(["cached", "downloaded"]).sum()
            )
            if not resources.empty
            else 0,
            "public_tabular_cells": len(frames["public_tabular_cells"]),
            "public_document_pages": len(frames["public_document_pages"]),
            "public_dashboard_records": len(frames["public_dashboard_records"]),
            "quantitative_artifacts": len(list(quantitative.glob("*.csv")))
            + len(list(quantitative.glob("*.parquet"))),
            "image_artifacts": len(list(settings.paths.images.glob("*.png"))),
        }
    )
    _write_json(summary, summary_path)
    artifacts = sorted(
        [
            *quantitative.glob("*.csv"),
            *quantitative.glob("*.parquet"),
            *settings.paths.images.glob("*.png"),
        ]
    )
    manifest = pd.DataFrame.from_records(
        [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "artifact_type": "image" if path.suffix == ".png" else "quantitative",
            }
            for path in artifacts
        ]
    )
    manifest_csv = _write_frame(manifest, root / "artifact_manifest.csv")
    _write_frame(manifest, root / "artifact_manifest.parquet")
    return {"root": root, "manifest": manifest_csv, "summary": summary_path}


def refresh_public_artifacts(settings: Settings) -> dict[str, Path]:
    frames = harvest_public_sources(settings)
    for name, frame in frames.items():
        _save_table(frame, settings.paths.quantitative, name)
    return _finalize_public_artifacts(settings, frames)


def finalize_public_artifacts(settings: Settings) -> dict[str, Path]:
    names = [
        "public_dataset_catalog",
        "public_repository_catalog",
        "public_source_resources",
        "public_dashboard_status",
        "public_dashboard_records",
        "public_tabular_cells",
        "public_document_pages",
    ]
    frames = {name: _read(settings.paths.quantitative, name) for name in names}
    return _finalize_public_artifacts(settings, frames)


def refresh_public_repository_artifacts(settings: Settings) -> dict[str, Path]:
    replacements = harvest_public_repositories(settings)
    names = [
        "public_data_go_search_status",
        "public_dataset_catalog",
        "public_repository_catalog",
        "public_repository_search_status",
        "public_landing_page_status",
        "public_source_resources",
        "public_dashboard_status",
        "public_dashboard_records",
        "public_extraction_status",
        "public_tabular_tables",
        "public_tabular_cells",
        "public_document_pages",
    ]
    frames = {name: _read(settings.paths.quantitative, name) for name in names}
    for name, replacement in replacements.items():
        if name in {"public_repository_catalog", "public_repository_search_status"}:
            frames[name] = replacement
        else:
            current = frames[name]
            retained = current.loc[current["source"].ne("zenodo")]
            frames[name] = pd.concat([retained, replacement], ignore_index=True)
        _save_table(frames[name], settings.paths.quantitative, name)
    return _finalize_public_artifacts(settings, frames)


def refresh_open_dashboard_artifacts(settings: Settings) -> dict[str, Path]:
    frames = harvest_open_dashboards(settings)
    documents, sources = harvest_kemkes(settings, requests.Session())
    weekly, ebs = parse_skdr(documents)
    frames.update(
        {
            "kemkes_document_text": documents,
            "kemkes_source_manifest": sources,
            "skdr_weekly_indicators": weekly,
            "skdr_ebs_indicators": ebs,
        }
    )
    for name, frame in frames.items():
        _save_table(frame, settings.paths.quantitative, name)
    return {name: settings.paths.quantitative / f"{name}.csv" for name in frames}

from __future__ import annotations

import argparse
from pathlib import Path

from dengue_st_diagnostics.config import Settings
from dengue_st_diagnostics.refresh import (
    finalize_public_artifacts,
    refresh_open_dashboard_artifacts,
    refresh_public_artifacts,
    refresh_public_repository_artifacts,
    refresh_surveillance_artifacts,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--config", type=Path, default=Path("config/default.json"))
    value.add_argument("--output", type=Path)
    value.add_argument("--input", type=Path, action="append", default=[])
    value.add_argument("--skip-downloads", action="store_true")
    value.add_argument("--skip-literature", action="store_true")
    value.add_argument("--refresh-surveillance", action="store_true")
    value.add_argument("--refresh-public", action="store_true")
    value.add_argument("--finalize-public", action="store_true")
    value.add_argument("--refresh-public-repositories", action="store_true")
    value.add_argument("--refresh-open-dashboards", action="store_true")
    value.add_argument("--lakehouse-init", action="store_true")
    value.add_argument("--lakehouse-load", action="store_true")
    value.add_argument("--lakehouse-export", action="store_true")
    value.add_argument("--lakehouse-enrichment-load", action="store_true")
    value.add_argument("--advanced-diagnostics", action="store_true")
    value.add_argument("--enrich-public-tables", action="store_true")
    return value


def main() -> int:
    arguments = parser().parse_args()
    settings = Settings.load(arguments.config, arguments.output)
    if arguments.enrich_public_tables:
        from dengue_st_diagnostics.enrichment import run

        run(settings)
        return 0
    if arguments.advanced_diagnostics:
        from dengue_st_diagnostics.advanced import run_advanced_diagnostics

        run_advanced_diagnostics(settings)
        return 0
    if arguments.refresh_surveillance:
        refresh_surveillance_artifacts(settings)
        return 0
    if arguments.refresh_public:
        refresh_public_artifacts(settings)
        return 0
    if arguments.finalize_public:
        finalize_public_artifacts(settings)
        return 0
    if arguments.refresh_public_repositories:
        refresh_public_repository_artifacts(settings)
        return 0
    if arguments.refresh_open_dashboards:
        refresh_open_dashboard_artifacts(settings)
        return 0
    if (
        arguments.lakehouse_init
        or arguments.lakehouse_load
        or arguments.lakehouse_enrichment_load
        or arguments.lakehouse_export
    ):
        from dengue_st_diagnostics.lakehouse import Lakehouse

        lakehouse = Lakehouse(settings)
        if arguments.lakehouse_init:
            lakehouse.initialize()
        if arguments.lakehouse_load:
            lakehouse.initialize()
            lakehouse.load()
        if arguments.lakehouse_enrichment_load:
            lakehouse.initialize()
            lakehouse.load_enrichment()
        if arguments.lakehouse_export:
            lakehouse.export()
        return 0
    from dengue_st_diagnostics.pipeline import run_pipeline

    run_pipeline(
        settings,
        input_paths=arguments.input,
        download_sources=not arguments.skip_downloads,
        harvest_literature=not arguments.skip_literature,
    )
    return 0

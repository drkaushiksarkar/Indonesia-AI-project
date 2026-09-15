# Indonesia Vector-Borne Disease Data Lake and Diagnostics

This repository contains the reproducible Python and SQL implementation for harvesting, normalizing, preserving and diagnosing public Indonesian dengue and malaria data across native spatial and temporal grains.

The code supports national, provincial, district, subdistrict and facility-level observations at weekly, monthly, annual and snapshot frequencies when those grains are present in the source. It preserves the source grain and provenance rather than forcing all observations into one analytical table.

## Repository scope

- Public-source harvesting for surveillance portals, document libraries, public dashboards, repositories and APIs
- PostgreSQL bronze, silver, gold and metadata layers
- Native-grain surveillance facts with lineage and source metadata
- Temporal, spatial, spatiotemporal and nonlinear diagnostics
- Rolling-origin univariate forecasting, outbreak detection and risk-stratification evaluation
- Static executive-facing analytical reports and figures in `docs/`

Raw data, database dumps, exports, generated quantitative outputs and credentials are intentionally excluded from version control.

## Environment

Python 3.11 to 3.13 and PostgreSQL are required. The project uses `uv.lock` for a reproducible Python environment.

```bash
uv sync --extra dev
uv run ruff check src tests
uv run pytest
```

## Main commands

```bash
uv run dengue-st-diagnostics --config config/default.json
uv run dengue-st-diagnostics --config config/default.json --lakehouse-init
uv run dengue-st-diagnostics --config config/default.json --lakehouse-load
uv run dengue-st-diagnostics --config config/default.json --lakehouse-export
uv run dengue-st-diagnostics --config config/default.json --advanced-diagnostics
uv run dengue-univariate-frontier
```

Set `INDONESIA_VECTOR_LAKE_DSN` before database operations. SATUSEHAT FHIR harvesting remains disabled unless valid `SATUSEHAT_CLIENT_ID` and `SATUSEHAT_CLIENT_SECRET` environment variables are supplied.

## Analytical outputs

Open `docs/index.html` locally for the report portal. It contains the validated predictability report and a curated gallery of intact analytical figures. The same folder is ready to publish through GitHub Pages after its public-release status has been approved.

## Data distribution

The complete data lake and client delivery archive are separate from this code repository. The 2026-08-30 dated archive failed content-integrity review and must not be distributed. Build and validate a fresh checksummed delivery from hydrated files and the live database, then use an access-controlled object store for distribution. See `PUBLISHING.md` for the release sequence.

## License and source rights

No project license is asserted in this repository. Select a software license and confirm the redistribution terms of each source before making the repository or any harvested data public.

## Bronze enrichment audit

See [ENRICHMENT_AUDIT.md](ENRICHMENT_AUDIT.md) for measured extraction gaps and the new source-preserving province/year panel. The panel is available through `python -m dengue_st_diagnostics.research_panel`; its screened gold features are for retrospective experiments.

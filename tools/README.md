# Disease mining tools

Run from the repository root. Outputs default to `../outputs/extensive_mining`; the parent directory must exist. Python dependencies: `uv sync --extra dev --extra mining`. PDF extraction needs Poppler (`pdftotext`); optional image-chart OCR needs Tesseract. No climate files are collected by the case-reference tool.

## Collection

- `mine_catalogs.py`, `mine_more_portals.py`, `mine_observed_portals.py`, `retry_catalog_pages.py`: public catalogue discovery, pagination and receipts. Inspect failures; a CKAN probe does not prove a website has no data.
- `mine_downloads.py`: candidate disease resources and reuse of existing bronze resource-cache paths; requires the local PostgreSQL lake. Files are capped at 100 MB each.
- `index_mined_files.py`: spreadsheet and PDF text index; identified personal health registers are excluded from indexing.
- `mine_reference_case_data.py`: pinned provincial dengue RDS/CSV and the public Yogyakarta case workbook.
- `mine_semarang.py`: public disease chart snapshots, with source labels preserved.
- `mine_inhu_bulletins.py`, then `download_inhu_bulletins.py`: official annual archives, cached landing pages and public PDF/Drive embeds. The downloader reads a snapshot of available HTML: rerun after landing-page collection completes.
- `mine_national_bulletins.py`: published Ministry bulletins.
- Research discovery scripts retain source links, successful files and failures. They do not request restricted data or send messages.
- `mine_skdr_nationwide.py` is **held for source quality failure** in this collection. Do not treat its development dashboard chart slots as validated surveillance. See the audit and `skdr_nationwide/collection_status.json`.

Individual collectors are resumable to varying degrees; read the script before rerunning. Some discovery scripts refresh their manifests. Raw output and credentials must remain outside Git.

## Extraction and publication

Use the existing collected raw files and manifests:

```bash
export PYTHONPATH=src
uv run python tools/parse_portal_series.py
uv run python tools/parse_malaria_papers.py
uv run python tools/parse_district_papers.py
uv run python tools/parse_inhu_bulletins.py
uv run python tools/parse_national_bulletins.py
uv run python tools/summarize_skdr.py
uv run python -m dengue_st_diagnostics.extensive_mining
uv run python tools/compare_mining_baseline.py
uv run python -m dengue_st_diagnostics.extensive_mining --publish
```

The first normalization creates the comparison input. The second applies the saved baseline comparison and excludes otherwise eligible differing counts from gold. Refresh the comparison whenever source or baseline data changes. Comparison is deliberately conservative and name matching is incomplete; unmatched is not synonymous with new.

`--root` changes the normalization input/output root; `--database` selects the PostgreSQL database (default `indonesia_vector_lake`). The mining scripts currently use the documented default output root. The new module does not require API credentials.

`ocr_bulletin_candidates.py` renders image-based charts and saves OCR for review. OCR is intentionally not merged into observations. PDFium rendering runs serially because its rendering API is not thread-safe.

Tests and lint:

```bash
uv run ruff check src tests tools
uv run pytest
```

Do not sum report snapshots, mixed case definitions, different geographic levels, or research cohorts with surveillance totals. See [MINING_AUDIT.md](../MINING_AUDIT.md) for current counts, exclusions and access gaps.

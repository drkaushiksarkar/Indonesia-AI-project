# Bronze-to-model enrichment audit — 2026-09-15

## Findings from the live PostgreSQL lake

The existing public-table enrichment query accepts only disease-titled `data.go.id` tables. It excludes research-repository tables and has no general recovery path for wide year columns, multirow headers, or delimiter-damaged tables.

- Public tabular bronze: **1,962,004 nonempty cells** in **2,277 tables**.
- Source split: Zenodo 1,391,144 cells; data.go.id 440,919; JCU 129,941.
- Existing government enrichment audit: 1,067 tables assessed; 456 eligible and 611 requiring review. The 611 review tables contain 39,316 cells. Some nominally eligible tables also produced no observations.
- Existing gold native observations: 51,681 across eight source datasets, including 14,709 prior public-enrichment observations.

Cells, measurements and model examples are different units. The cell count is not a denominator for a valid extraction-success percentage. Research cells include global vector occurrences, survey records, metadata and unrelated material; they must not all become Indonesian case counts.

## Implemented: aggregate province/year research panel

Recovered Zenodo record **20034615**, `Data DBD Provinsi 2017-2024 + Fitur Baru.csv`, directly from existing bronze cells. Source schema is pinned and checked before parsing. The original 35 province labels resolve to 34 after the explicit Bangka Belitung abbreviation mapping. No other geography is inferred.

| Stage | Count |
|---|---:|
| Bronze cells, including headers | 6,209 |
| Province/year records, 2017–2024 | 272 |
| Expected measurements, 21 fields per record | 5,712 |
| Measurements passing cell and row checks | 2,992 |
| Measurements requiring review | 2,720 |
| Complete lag-one-year modeling examples after checks | 86 |

Seventy values are absent or out of range: 68 PVT values, one environmental-health-quality value, and one TFS value. Missing cells have no bronze cell ID; existing cells retain their original IDs and coordinates. Silver keeps numeric source values on rows failing consistency checks; cell-invalid values remain null with a review flag, while original bronze remains intact.

Further row checks found:

- 122 province/year rows with reported incidence differing from cases / population × 100,000 by more than 1.
- 16 rows with case-fatality percentages differing from deaths / cases × 100 by more than 0.1 percentage point.
- No discrepancies between male + female and total population exceeding one person; no deaths exceeding cases; no zero populations.

These thresholds are review tolerances, not proof that a particular source field is wrong. Different denominators or transcription errors may explain discrepancies. For example, Bali 2017 reports 123 cases, population 4,245,294 and incidence 28.97; the supplied counts imply approximately 2.90. No automatic correction is made. Row-level failures quarantine all associated measurements from feature construction. The failure categories overlap; do not sum them as distinct rows.

### Database tables

- `silver.research_panel_observation`: source measurements, status, units, source headers, coordinates and bronze linkage.
- `gold.research_panel_features`: current-year case target with 12 complete prior-year predictors. Predictors include prior cases/deaths, population, density, area, facilities, sanitation, drinking water, poverty, Gini and HDI.

The tables are separate from existing native observations and forecasting snapshots. They are not appended to existing series, preventing silent overlap with other sources. Gold examples are labeled `retrospective_only`; source publication dates are unverified and the source's 34-province geography is not harmonized. A lag reduces same-year leakage but does not establish historical availability. Validate boundaries and availability before operational modeling. No new model has been fitted and no forecast improvement is claimed.

### Reproduce

From the repository root, with the existing lake accessible:

```bash
uv sync --frozen --extra dev
uv run python -m dengue_st_diagnostics.research_panel --database indonesia_vector_lake
# Publish only this source to the two dedicated tables in one transaction:
uv run python -m dengue_st_diagnostics.research_panel --database indonesia_vector_lake --publish
```

Outputs are written to ignored `outputs/research_panel/`. Re-running publication refreshes only this source's rows, including removed rows. Other sources and existing gold tables are preserved. The extraction requires the preserved bronze lake; the repository does not distribute the database.

## Prioritized remaining extraction work

1. **Historical dengue counts:** Zenodo `11451183` contains 946 data rows plus a header, with explicit period starts/ends, counts, geography and case-definition fields. A dedicated parser must preserve cumulative flags, age ranges, diagnosis certainty and the difference between deaths and cases. Reconcile against existing canonical/DengueNet series before promotion; the rows are candidates, not proven net additions.
2. **Government wide and malformed tables:** one inspected mixed-disease table stores `WABAH/PENYAKIT;2022;2023` in a single CSV cell, while its spreadsheet representation has proper year columns. Prefer the intact spreadsheet; add a wide-year parser and deduplicate representations rather than counting both.
3. **Multirow headers and PDF tables:** prioritize aggregate tables in the 611-table review queue. Keep header detection, geography and time checks; do not lower confidence thresholds globally. Some large tables are metadata, not observations.
4. **Vector covariates:** the global Aedes compendium accounts for 920,502 cells. Filter to Indonesia, validate coordinate systems and dates, deduplicate and aggregate at supported geographic/time scales before a covariate join.
5. **Research surveys and JCU data:** review study populations and variable definitions before extraction. These are not interchangeable with aggregate surveillance outcomes.
6. **Coverage gaps after extraction:** measure missing province/month/year combinations before downloading more data. Reconcile datasets by source, definition and grain, and track what actually adds unique information.

## Evidence

The companion `notebooks/bronze_enrichment_audit.ipynb` contains repeatable read-only queries and panel checks. Source parsing and feature tests live in `tests/test_research_panel.py`. This audit covers stored bronze and the inspected source; it is not a complete review of all raw files or newly available external data.

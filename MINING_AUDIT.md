# Dengue and malaria mining audit — 15 September 2026

## Outcome

This collection focuses on dengue and malaria. No climate variables are downloaded by the new case-reference collector or included in the new forecasting features.

The sweep inspected 119 saved catalogue results, with disease packages found in 49 catalogues (1,223 packages). It resolved 1,116 candidate resource URLs: 1,020 available resources, representing 899 distinct file hashes; 96 failed. These are retrieval counts, including previously cached resources, annual tables, duplicate publications and service indicators—not counts of new time series.

Additional collection retrieved 97 national Ministry bulletins (95 distinct files), 100 Indragiri Hulu weekly bulletins, public Semarang disease charts and published research case tables. Indragiri Hulu's three annual archives exposed 126 landing pages; 107 were accessible after retry, yielding 100 distinct PDFs through official embeds and public Google Drive links.

## Case coverage recovered

| Source | Disease and grain | Observed coverage | Limitation |
|---|---|---|---|
| Public provincial research file | Dengue, 34 historical provinces plus national, monthly | 2010–2023; 168 months per series | Substantial overlap with existing OpenDengue; historical boundaries and zero semantics require review |
| Central Java portal matrices | DBD cases and deaths, 35 districts/cities, monthly | 2022 Jan–Oct, 2023 Jan–Nov, 2024 Jan; 770 selected case cells | 22 months per district, with gaps; earlier report revisions retained separately |
| Semarang official dashboard | Dengue and malaria, municipality, monthly; dengue weekly | Monthly 2019 onward; weekly dengue 2021 onward in returned records | Current years partial; weekly calendar and missing week labels preserved |
| Malang facility files | Dengue and malaria, puskesmas and kelurahan, monthly | Selected Arjowinangun, Pandanwangi and Mojolangu files, 2022–2025 | Coverage varies; adds subunit detail and previously missed files, not all periods for every facility |
| Belitung portal | Dengue, district, monthly | 2018–2019, 24 months | 20 monthly counts disagree with existing observations; held for reconciliation |
| Penajam Paser Utara portal | Malaria, district, monthly | 2025, 12 months | Short history; earlier partial release revised |
| Tulang Bawang Barat paper | Dengue, district, monthly | 2019–2023, 60 months | Explicit zeros held for completeness review; one narrative/table discrepancy |
| Hanura paper | Malaria, puskesmas, monthly | 2019–2020, 24 months, including microscopy/RDT counts | March 2020 total 476 differs from 90+336 diagnostic counts; retained without repair |
| Weoe paper | Malaria-positive research sample, facility, monthly | 2019, 12 months | 815 of 860 tests eligible for study; not complete facility surveillance |
| Yogyakarta Wolbachia study | Dengue, two study catchments, monthly | 2006 Jan–2019 Sep, 165 months | Study areas are not municipality totals; intervention affects transmission |
| Indragiri Hulu official bulletins | Suspected dengue and confirmed malaria, district, weekly | Dengue records span 2024-W01–2026-W22; malaria 2024-W01–W43 | Gaps remain; explicit zero reports retained; not every week available |
| Indragiri Hulu narratives | Suspected dengue, nine named puskesmas, weekly | Sparse named counts | Automated allocations reconcile to district totals; narrative review flag retained |
| National published SKDR bulletins | Suspected dengue and confirmed malaria, 38 provinces, cumulative snapshots | 3,132 deduplicated values, cutoffs spanning 2025-W16–2026-W34 | Year-to-date totals, **not weekly incident counts**; never summed or blindly differenced |

Dates above describe bounds, not guaranteed uninterrupted coverage. Provincial and district totals overlap; do not add them to facility or study-area counts.

## Source quality failure found

The public `surkarkes.kemkes.go.id/dashboard-skdr-dev` navigation exposed 561 geographical entries but only 554 distinct province/district filter combinations. Several different district names shared a filter code. The collection was stopped after 105 combined charts when Aceh Barat returned **256,211 suspected dengue cases and 155,778 malaria cases for 2025-W32**. The 2026 charts also returned zeros despite nonzero counts in published official bulletins.

All 20,265 chart slots remain quarantined with `source_validation_failed`. They do not enter forecasting features. This is not evidence of complete national weekly surveillance. The collector contains a source-quality hold for this saved collection. Source-native geography codes are namespaced rather than silently equated to BPS codes.

## Normalization and model candidates

Current dedicated outputs contain 37,130 source-preserving observations, of which 20,265 are the quarantined dashboard slots. The remaining 16,865 include cases, deaths, diagnostic measures, sex-specific rows, derived all-sex totals, cumulative snapshots and report revisions. This is **not** a count of independent new cases or entirely new observations relative to the old lake.

Gold contains **4,624 monthly feature rows with complete required history**, using calendar lags 1, 2, 3, 6 and 12 and the mean of the preceding three months. These are experiment candidates, not a claim of operational forecasting readiness. The largest component overlaps existing provincial data. Missing months are not compressed, interpolated or zero-filled. Study subsets, unresolved calendars, cumulative totals, source failures, arithmetic discrepancies and identified cross-source conflicts are excluded from these features.

The baseline comparison found 4,948 same-valued provincial case cells and 143 differing provincial cells, plus the 20 Belitung conflicts. Of the differing cells, 145 had otherwise passed the initial extraction screen and are now marked `cross_source_review`; the others already carried review flags. Unmatched names are not counted as proven new records. No cross-source totals are summed.

Tables in local database `indonesia_vector_lake`:

- `bronze.extensive_mining_resource`: 1,464 retrieval/provenance records.
- `silver.extensive_mining_observation`: 37,130 observations with source locator, raw file hash and quality fields.
- `gold.extensive_mining_features`: 4,624 complete-history candidate rows.
- `gold.extensive_mining_readiness`: 389 source/location/disease/frequency series assessments; this is not a count of distinct geographic units.

Existing native historical tables remain intact. Dedicated tables are replaced transactionally on a rerun; resource receipts are upserted.

## Remaining gaps and access leads

- The public sources do **not** provide adequate long weekly/monthly histories for every Indonesian district or puskesmas. Malaria facility histories are particularly sparse. National open coverage must not be inferred from catalogue counts.
- 32 image-based or incompletely parsed national charts have rendered images and OCR text in the review queue. OCR is not automatically promoted to case values.
- 19 Indragiri Hulu landing pages remained inaccessible after retry. Some retrieved bulletins did not expose an unambiguous machine-readable target case table.
- The paper describing monthly dengue for 514 regencies/cities in 2010–2020 explicitly states that its authors cannot share the dataset: [published study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11122138/).
- National district malaria monthly data for 2010–2019 are described in a study but require an authorized data request: [study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11881261/).
- Papua clinic malaria 2019–2023 and Kulon Progo/Magelang facility histories are access leads, not acquired raw time series: [Papua study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12135496/), [facility study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11628951/).
- IMERI provincial dengue 2019–2024 is restricted to the GFID research project: [catalogue record](https://aida.informatics.buu.ac.th/dataset/indonesia_dengue_2019-2024_monthly).
- Mimika public supplementary spreadsheets contain multi-year period aggregates, not the monthly series described in the paper. The Bandung supplement has 30 areas × 12 month labels without year identifiers; no fictitious multi-year panel was constructed.
- A named patient register encountered in a public portal was excluded from the text index and normalized outputs. Raw downloads remain local and are not committed to GitHub.

No outreach or access requests were sent. The next major gain requires authorized aggregate exports from SKDR/e-SISMAL and provincial/district health offices, including facility identifiers, case definitions, reporting completeness and revision dates.

## Verification and use

37 tests pass, including native week labels, explicit zero versus missing data, two-week bulletin year boundaries, invalid counts, conflict holds, missing-month lags and source-quality exclusion. PDF case tables for Hanura, Weoe, Tulang Bawang Barat, selected Malang records and the example puskesmas narrative were visually inspected. Province chart extraction uses explicit labels and printed cutoff weeks. OCR candidates remain held.

Local artifacts live under `../outputs/extensive_mining/` relative to this repository:

- `normalized/observations.parquet` — all observations and flags.
- `normalized/case_series_with_quality_flags.csv` — all-sex case series excluding archived duplicate releases and the failed dashboard.
- `normalized/forecast_candidates.parquet` — complete-history candidate rows matching gold.
- `normalized/series_readiness.csv` — individual coverage and gap checks.
- `normalized/baseline_comparison.csv` — baseline matching evidence.
- `national_bulletins/ocr_manifest.json` — image-chart review queue.
- `MINING_AUDIT.md` — this report copied alongside the data.

See [tools/README.md](tools/README.md) for reruns. Review definitions, boundary changes, revisions, zero semantics and intended prediction horizon before training. A count of 36 monthly periods alone is not proof of model adequacy.

## Primary source links

[Semarang dashboard](https://lekminkes.dinkes.semarangkota.go.id/), [Central Java portal](https://data.jatengprov.go.id/), [Malang portal](https://data.malangkota.go.id/), [national bulletin archive](https://surveilans.kemkes.go.id/publikasi), [Indragiri Hulu 2024 archive](https://dinkes.inhukab.go.id/buletin-skdr-tahun-2024/), [2025 archive](https://dinkes.inhukab.go.id/buletin-skdr-tahun-2025/), [2026 archive](https://dinkes.inhukab.go.id/buletin-skdr-tahun-2026/), [Yogyakarta case workbook](https://doi.org/10.6084/m9.figshare.12199688), [Hanura table](https://ejurnalmalahayati.ac.id/index.php/kesehatan/article/download/18958/pdf), [Weoe table](https://journal.ugm.ac.id/bik/article/download/55908/31370), [Tulang Bawang Barat table](https://ejurnalmalahayati.ac.id/index.php/medika/article/viewFile/17967/pdf).

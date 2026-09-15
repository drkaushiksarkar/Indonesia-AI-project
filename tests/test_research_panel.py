import pandas as pd
import pytest

from dengue_st_diagnostics.research_panel import FIELDS, build_features, extract


def cells():
    headers = ["Tahun", "Provinsi", *FIELDS]
    result = []
    data = [headers]
    for year, province, count in [
        (2017, "Kep. Bangka Belitung", 10),
        (2018, "Kepulauan Bangka Belitung", 20),
    ]:
        values = dict.fromkeys(FIELDS, "10")
        values.update(
            {
                "Laki-laki": "500",
                "Perempuan": "500",
                "Total Penduduk": "1000",
                "Jumlah Kasus": str(count),
                "Jumlah Kasus Meninggal": str(count // 10),
                "Incidence Rate per 100.000 Penduduk": str(count * 100),
                "Case Fatality Rate (%)": "10",
            }
        )
        data.append([str(year), province, *values.values()])
    for r, values in enumerate(data, 1):
        for c, value in enumerate(values, 1):
            result.append(
                (
                    r * 100 + c,
                    {
                        "row_number": str(r),
                        "column_number": str(c),
                        "value": value,
                        "source_url": "https://example.test",
                    },
                )
            )
    return result


def test_alias_lineage_and_lagged_features():
    observations, issues = extract(cells())
    assert issues.empty
    assert len(observations) == 42
    assert observations.province.nunique() == 1
    assert observations.bronze_record_id.notna().all()
    features = build_features(observations)
    assert len(features) == 1
    assert features.iloc[0].target_cases == 20
    assert features.iloc[0].predictors["cases_lag1"] == 10
    assert features.iloc[0].predictors["population_lag1"] == 1000
    assert "pvt_source_defined_lag1" not in features.iloc[0].predictors


def test_missing_year_is_not_bridged():
    observations, _ = extract(cells())
    observations.loc[observations.year.eq(2018), "year"] = 2019
    assert build_features(observations).empty


def test_invalid_values_are_retained_for_review_not_predictors():
    source = cells()
    column = ["Tahun", "Provinsi", *FIELDS].index("Sanitasi Layak (%)") + 1
    for _, payload in source:
        if payload["row_number"] == "2" and int(payload["column_number"]) == column:
            payload["value"] = "125"
    observations, issues = extract(source)
    assert len(issues) == 1
    bad = observations[observations.quality_status.eq("review_required")].iloc[0]
    assert pd.isna(bad.value)
    assert build_features(observations).empty


def test_duplicate_coordinates_and_changed_schema_fail_closed():
    source = cells()
    with pytest.raises(ValueError, match="Duplicate source coordinate"):
        extract([*source, source[-1]])
    with pytest.raises(ValueError, match="Unexpected panel schema"):
        extract(source[1:])


def test_inconsistent_outcomes_are_quarantined():
    source = cells()
    column = ["Tahun", "Provinsi", *FIELDS].index("Jumlah Kasus") + 1
    for _, payload in source:
        if payload["row_number"] == "3" and int(payload["column_number"]) == column:
            payload["value"] = "2"
    observations, issues = extract(source)
    assert "incidence_disagrees_with_cases_population" in set(issues.reason)
    assert observations[observations.year.eq(2018)].quality_status.eq("review_required").all()
    assert build_features(observations).empty

import pandas as pd
import pytest

from dengue_st_diagnostics.extensive_mining import (
    deduplicate,
    monthly_features,
    observation,
    semarang_records,
)


def test_week_labels_are_not_compressed_or_zero_filled():
    item = {
        "variable": "dbd_mingguan",
        "year": 2025,
        "url": "https://example.org",
        "path": "raw.json",
    }
    payload = {
        "status": True,
        "title": "DBD",
        "mingguan": ["1", "3"],
        "data": [
            {"name": "Penderita Laki-laki", "data": [2, 4]},
            {"name": "Penderita Perempuan", "data": [1, 5]},
        ],
    }
    records = semarang_records(payload, item)
    totals = {r["period"]: r["value"] for r in records if r["sex"] == "all"}
    assert totals == {"2025-W01": 3, "2025-W03": 9}
    assert all(r["quality"] == "calendar_review" for r in records)


def test_mismatched_week_labels_rejected():
    with pytest.raises(ValueError, match="week labels"):
        semarang_records(
            {"status": True, "data": [{"name": "cases", "data": [1, 2]}], "mingguan": ["1"]},
            {"year": 2024, "variable": "dbd_mingguan"},
        )


def test_calendar_lag_does_not_jump_over_missing_month():
    frame = pd.DataFrame(
        [
            observation("test", "123", "Town", "admin2", p, v)
            for p, v in [("2023-01", 7), ("2023-03", 9), ("2023-04", 11)]
        ]
    )
    features = monthly_features(frame).set_index("period")
    assert pd.isna(features.loc["2023-03", "cases_lag1"])
    assert features.loc["2023-04", "cases_lag1"] == 9
    assert pd.isna(features.loc["2023-04", "cases_rolling3_prior"])


def test_conflicting_reports_are_quarantined_not_summed():
    records = [observation("test", "123", "Town", "admin2", "2023-01", v) for v in [7, 7, 9]]
    frame, conflicts = deduplicate(pd.DataFrame(records))
    assert len(frame) == 2
    assert len(conflicts) == 1
    assert set(frame.quality) == {"conflict"}


def test_non_count_rejected():
    for value in [-1, 2.5, float("nan")]:
        with pytest.raises(ValueError, match="Invalid count"):
            observation("test", "123", "Town", "admin2", "2023-01", value)


def test_source_quality_failure_excluded_from_features():
    good = observation("test", "123", "Town", "admin2", "2023-01", 7)
    bad = observation(
        "test", "123", "Town", "admin2", "2023-02", 999999, quality="source_validation_failed"
    )
    features = monthly_features(pd.DataFrame([good, bad]))
    assert features.period.tolist() == ["2023-01"]


def test_bulletin_cases_do_not_use_alert_column_or_missing_disease_zero():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).parents[1] / "tools" / "parse_inhu_bulletins.py"
    spec = importlib.util.spec_from_file_location("parse_bulletin", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = "MINGGU EPIDEMIOLOGI KE-15 TAHUN 2026\nPENYAKIT KASUS ALERT KLB\n2 Suspek Dengue 5 2 0\n"
    rows = module.parse_table(
        text, {"url": "https://example.org/report.pdf", "path": "raw.pdf", "sha256": "test"}
    )
    assert len(rows) == 1
    assert rows[0]["value"] == 5
    assert rows[0]["disease"] == "dengue"
    assert rows[0]["period"] == "2026-W15"


def test_only_held_data_produces_no_gold_features():
    held = observation(
        "test", "123", "Town", "admin2", "2023-01", 999999, quality="source_validation_failed"
    )
    assert monthly_features(pd.DataFrame([held])).empty


def test_two_week_bulletin_preserves_explicit_zero_and_previous_year():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "parse_bulletin", Path(__file__).parents[1] / "tools/parse_inhu_bulletins.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = "MINGGU KE 1 TAHUN 2024\nDATA SKDR PENYAKIT POTENSIAL WABAH 2 MINGGU TERAKHIR\nNo Penyakit M-52 M-1\n2 Malaria Konfirmasi 0 0\n3 Suspek Dengue 4 2\n"
    rows = module.parse_table(
        text, {"url": "https://example.org/report.pdf", "path": "raw.pdf", "sha256": "test"}
    )
    assert {(r["disease"], r["period"], r["value"]) for r in rows} == {
        ("malaria", "2023-W52", 0),
        ("malaria", "2024-W01", 0),
        ("dengue", "2023-W52", 4),
        ("dengue", "2024-W01", 2),
    }

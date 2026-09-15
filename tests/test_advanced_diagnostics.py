from __future__ import annotations

import numpy as np
import pandas as pd

from dengue_st_diagnostics.advanced.series import build_series_catalog
from dengue_st_diagnostics.advanced.storage import export_frames, false_discovery_rate
from dengue_st_diagnostics.advanced.temporal import _emd


def test_series_catalog_preserves_upstream_source_and_subunit() -> None:
    rows = []
    for upstream, subunit in (("source_a", None), ("source_b", None), ("source_a", "Village")):
        for index, period in enumerate(pd.date_range("2024-01-01", periods=8, freq="MS")):
            rows.append(
                {
                    "observation_id": len(rows) + 1,
                    "source_id": "derived",
                    "source_dataset_id": 1,
                    "source_dataset_name": "cases",
                    "upstream_source": upstream,
                    "disease": "dengue",
                    "metric": "cases",
                    "unit": "cases",
                    "spatial_resolution": "adm4" if subunit else "facility",
                    "spatial_unit_type": "kelurahan" if subunit else "puskesmas",
                    "temporal_resolution_normalized": "month",
                    "admin_0": "Indonesia",
                    "admin_1": "Province",
                    "admin_2": "District",
                    "admin_3": None,
                    "facility_name": "Clinic",
                    "facility_type": "puskesmas",
                    "source_subunit": subunit,
                    "source_aggregation_level": "kelurahan" if subunit else "puskesmas",
                    "sex": None,
                    "age_group": None,
                    "species": None,
                    "case_definition": "reported",
                    "confirmation_status": "unknown",
                    "period_start": period,
                    "period_end_effective": period + pd.offsets.MonthEnd(0),
                    "value": float(index),
                }
            )
    frame = pd.DataFrame.from_records(rows)
    frame["location_name"] = ""
    from dengue_st_diagnostics.advanced.series import _identifier, _location

    frame["location_name"] = frame.apply(_location, axis=1)
    from dengue_st_diagnostics.advanced.series import PANEL_DIMENSIONS, SERIES_DIMENSIONS

    frame["series_id"] = frame.apply(
        lambda row: _identifier("series", {key: row.get(key) for key in SERIES_DIMENSIONS}),
        axis=1,
    )
    frame["panel_id"] = frame.apply(
        lambda row: _identifier("panel", {key: row.get(key) for key in PANEL_DIMENSIONS}),
        axis=1,
    )
    catalog, inputs = build_series_catalog(frame)
    assert len(catalog) == 3
    assert catalog["regular_time"].all()
    assert len(inputs) == len(frame)


def test_false_discovery_rate_is_monotone_and_bounded() -> None:
    frame = pd.DataFrame(
        {
            "method_id": ["test"] * 4,
            "statistic_name": ["score"] * 4,
            "p_value": [0.001, 0.01, 0.03, 0.8],
        }
    )
    result = false_discovery_rate(frame)
    assert result["q_value"].between(0, 1).all()
    assert np.all(np.diff(result["q_value"]) >= 0)


def test_emd_reconstructs_input() -> None:
    time = np.linspace(0, 8 * np.pi, 128)
    values = np.sin(time) + 0.3 * np.sin(4 * time) + np.linspace(0, 1, len(time))
    imfs, residue = _emd(values)
    reconstructed = np.sum(imfs, axis=0) + residue
    assert imfs
    assert np.allclose(reconstructed, values)


def test_export_frames_serializes_empty_nested_values(tmp_path) -> None:
    frame = pd.DataFrame({"method_id": ["a", "b"], "parameters": [{}, {}]})
    paths = export_frames(tmp_path, {"methods": frame})
    restored = pd.read_parquet(paths[0])
    assert restored["parameters"].tolist() == ["{}", "{}"]

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from dengue_st_diagnostics.diagnostics import (
    diagnose_quality,
    diagnose_spatial,
    diagnose_spatiotemporal,
    diagnose_temporal,
)
from dengue_st_diagnostics.harvest.public_sources import (
    _data_go_links,
    _data_go_package,
    _flatten,
)
from dengue_st_diagnostics.harvest.puskesmas import _parse_purbalingga, puskesmas_readiness
from dengue_st_diagnostics.harvest.surveillance import parse_skdr
from dengue_st_diagnostics.multiscale import build_multiscale, grain_inventory
from dengue_st_diagnostics.normalize import normalize_paths
from dengue_st_diagnostics.schema import canonicalize


def synthetic_cases() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]
    rows = []
    for location_index, location in enumerate(names):
        for time_index, date in enumerate(dates):
            seasonal = 15 + 8 * np.sin(2 * np.pi * (time_index - location_index) / 12)
            rows.append(
                {
                    "source": "synthetic",
                    "source_record_id": f"{location_index}_{time_index}",
                    "location_id": location,
                    "location_name": location,
                    "admin_level": "admin1",
                    "admin_0": "Indonesia",
                    "admin_1": location,
                    "admin_2": pd.NA,
                    "admin_3": pd.NA,
                    "period_start": date,
                    "period_end": date + pd.offsets.MonthEnd(0),
                    "temporal_resolution": "month",
                    "cases": max(0, round(seasonal + location_index * 2)),
                    "population": 100_000 + location_index * 10_000,
                    "case_definition": "reported",
                    "confirmation_status": "mixed",
                    "native_record": True,
                }
            )
    return canonicalize(pd.DataFrame.from_records(rows))


def synthetic_boundaries() -> dict[str, gpd.GeoDataFrame]:
    names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]
    geometries = [box(index % 3, index // 3, index % 3 + 1, index // 3 + 1) for index in range(6)]
    frame = gpd.GeoDataFrame(
        {
            "shapeName": names,
            "shapeID": [str(index) for index in range(6)],
            "geometry": geometries,
        },
        crs=4326,
    )
    return {"adm1": frame}


def test_diagnostics() -> None:
    cases = synthetic_cases()
    multiscale = build_multiscale(cases, ["month", "quarter", "year"])
    inventory = grain_inventory(multiscale)
    quality, coverage = diagnose_quality(cases, multiscale)
    temporal = diagnose_temporal(multiscale, 8)
    global_spatial, local_spatial, matches = diagnose_spatial(
        multiscale,
        synthetic_boundaries(),
        19,
        7,
    )
    spatiotemporal = diagnose_spatiotemporal(multiscale, local_spatial)
    assert not inventory.empty
    assert not quality.empty
    assert coverage["completeness"].notna().all()
    assert "wavelet" in set(temporal["method"])
    assert "global_moran" in set(global_spatial["method"])
    assert not local_spatial.empty
    assert matches["matched"].max() == 6
    assert not spatiotemporal["dtw_clusters"].empty
    assert not spatiotemporal["modeling_readiness"].empty


def test_local_normalization(tmp_path: object) -> None:
    path = tmp_path / "dbd_2023.csv"
    pd.DataFrame(
        {
            "Provinsi": ["Alpha", "Beta"],
            "Bulan": [1, 1],
            "Tahun": [2023, 2023],
            "Jumlah Kasus DBD": [10, 20],
            "Jumlah Penduduk": [1000, 2000],
        }
    ).to_csv(path, index=False)
    cases, status = normalize_paths([path])
    assert len(cases) == 2
    assert cases["cases"].sum() == 30
    assert status["status"].iloc[0] == "parsed"


def test_puskesmas_annual_parser(tmp_path: object) -> None:
    path = tmp_path / "purbalingga.csv"
    pd.DataFrame(
        {
            "Nama Kecamatan": ["Alpha"],
            "UPT Puskesmas": ["Beta"],
            "Jumlah Pasien Laki - laki": [3],
            "Jumlah Pasien Perempuan": [4],
            "Jumlah Pasien Meninggal Laki - laki": [1],
            "Jumlah Pasien Meninggal Perempuan": [0],
        }
    ).to_csv(path, index=False)
    records = pd.DataFrame.from_records(
        _parse_purbalingga(path, 2020, "https://example.test/source.csv", "resource")
    )
    readiness = puskesmas_readiness(records)
    assert records["cases"].iloc[0] == 7
    assert records["deaths"].iloc[0] == 1
    assert readiness["observed_periods"].iloc[0] == 1


def test_skdr_dynamic_indicator_order() -> None:
    text = (
        "Kasus Suspek Dengue Tahun 2025\n"
        "1000\n200\n150\n100\n80\n70\n60\n"
        "ISPA\nDiare Akut\nILI\nSuspek Demam Tifoid\nPnemonia\n"
        "Suspek Dengue\nMalaria Konfirmasi\nM-1\n"
    )
    documents = pd.DataFrame.from_records(
        [
            {
                "title": "Laporan Mingguan M45 2025",
                "text": text,
                "url": "https://example.test/report.pdf",
            }
        ]
    )
    weekly, ebs = parse_skdr(documents)
    dengue = weekly.loc[weekly["indicator"].eq("Suspek Dengue"), "cumulative_reports"]
    malaria = weekly.loc[weekly["indicator"].eq("Malaria Konfirmasi"), "cumulative_reports"]
    assert dengue.iloc[0] == 70
    assert malaria.iloc[0] == 60
    assert ebs.empty


def test_data_go_flight_parser() -> None:
    package = {
        "name": "malaria-test",
        "title": "Malaria Test",
        "resources": [{"id": "one", "format": "CSV", "url": "https://example.test/a.csv"}],
    }
    flight = f"0:{json.dumps(package)}"
    payload = json.dumps([1, flight])
    html = (
        "<html><body><a href='/dataset/dataset/malaria-test'>x</a>"
        "<span>1 Datasets Found</span>"
        f"<script>self.__next_f.push({payload})</script></body></html>"
    )
    parsed = _data_go_package(html)
    links, count = _data_go_links(html, "https://data.go.id")
    assert parsed["resources"][0]["id"] == "one"
    assert links == ["https://data.go.id/dataset/dataset/malaria-test"]
    assert count == 1


def test_dashboard_flattening() -> None:
    values = dict(_flatten({"summary": [{"cases": 3}, {"cases": 5}]}))
    assert values["summary[0].cases"] == 3
    assert values["summary[1].cases"] == 5

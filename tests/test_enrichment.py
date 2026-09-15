from dengue_st_diagnostics.enrichment import _deduplicate, _parse_table


def _cells(
    dataset_id: str,
    title: str,
    resource: str,
    sheet: str,
    values: list[list[str]],
) -> list[tuple[object, ...]]:
    result = []
    bronze_id = 1
    for row_number, row in enumerate(values, start=1):
        for column_number, value in enumerate(row, start=1):
            result.append(
                (
                    bronze_id,
                    "data.go.id",
                    dataset_id,
                    title,
                    "resource-id",
                    resource,
                    "https://example.test/data",
                    "",
                    sheet,
                    row_number,
                    column_number,
                    value,
                )
            )
            bronze_id += 1
    return result


def test_weekly_dengue_table_uses_organization_as_district() -> None:
    rows = _cells(
        "weekly-dengue",
        "Jumlah Pasien DBD Mingguan 2022",
        "Jumlah Pasien DBD Mingguan 2022",
        "Sheet1",
        [
            [
                "Category",
                "Penderita Laki-laki",
                "Penderita Perempuan",
                "Meninggal Laki-laki",
                "Meninggal Perempuan",
            ],
            ["1", "8", "9", "0", "1"],
            ["2", "21", "8", "0", "1"],
        ],
    )
    observations, audit = _parse_table(rows, {"weekly-dengue": "Kota Semarang"})
    assert audit["status"] == "eligible"
    assert len(observations) == 8
    assert {row["temporal_resolution"] for row in observations} == {"week"}
    assert {row["admin_2"] for row in observations} == {"Kota Semarang"}
    assert {row["metric"] for row in observations} == {"cases", "deaths"}
    assert {row["sex"] for row in observations} == {"male", "female"}


def test_puskesmas_table_retains_facility_grain() -> None:
    rows = _cells(
        "puskesmas-dengue",
        "Kasus DBD Kabupaten Tegal Tahun 2020",
        "Kasus DBD 2020",
        "Sheet1",
        [
            [
                "tahun",
                "nama_kecamatan",
                "puskesmas",
                "jumlah_kasus",
                "jumlah_kasus_meninggal",
            ],
            ["2020", "MARGASARI", "KESAMBI", "12", "1"],
        ],
    )
    observations, audit = _parse_table(rows, {})
    assert audit["status"] == "eligible"
    assert len(observations) == 2
    assert {row["spatial_resolution"] for row in observations} == {"facility"}
    assert {row["facility_name"] for row in observations} == {"KESAMBI"}
    assert {row["admin_3"] for row in observations} == {"MARGASARI"}
    assert {row["metric"] for row in observations} == {"cases", "deaths"}


def test_equivalent_source_representations_are_deduplicated() -> None:
    rows = _cells(
        "duplicate-dengue",
        "Kasus DBD Kabupaten Tegal Tahun 2020",
        "Kasus DBD 2020.csv",
        "csv",
        [["tahun", "nama_kecamatan", "jumlah_kasus"], ["2020", "MARGASARI", "12"]],
    )
    observations, _ = _parse_table(rows, {})
    duplicate = observations[0].copy()
    duplicate["resource_id"] = "alternate-resource"
    duplicate["sheet"] = "data"
    duplicate["upstream_bronze_record_id"] = 999
    result = _deduplicate([observations[0], duplicate])
    assert len(result) == 1
    assert result[0]["source_representation_count"] == 2

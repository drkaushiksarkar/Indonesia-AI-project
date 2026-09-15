"""Extract explicit aggregate monthly tables; flag internal arithmetic conflicts."""

import json
import re
from pathlib import Path

from dengue_st_diagnostics.extensive_mining import MONTHS, observation

ROOT = Path("../outputs/extensive_mining")
if __name__ == "__main__":
    rows = []
    for item in json.loads((ROOT / "literature/additional_manifest.json").read_text()):
        if item["status"] != "available" or not item["path"].endswith(".pdf"):
            continue
        text = Path(item["path"]).with_suffix(".txt").read_text()
        if "malahayati" in item["url"]:
            table = text.split("Tabel 1. Distribusi frekuensi Kasus Malaria")[1].split("Rata-Rata")[
                0
            ]
            for match in re.finditer(
                r"^\s*([A-Za-z]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", table, re.M
            ):
                month = MONTHS.get(match[1].lower())
                if not month:
                    continue
                for year, offset in [(2019, 2), (2020, 5)]:
                    cases, micro, rdt = [int(match[i]) for i in range(offset, offset + 3)]
                    for metric, value in [
                        ("cases", cases),
                        ("microscopy_positive", micro),
                        ("rdt_positive", rdt),
                    ]:
                        rows.append(
                            observation(
                                "hanura_paper",
                                "1809:puskesmas:Hanura",
                                "Hanura",
                                "facility",
                                f"{year}-{month:02}",
                                value,
                                disease="malaria",
                                metric=metric,
                                url=item["url"],
                                path=item["path"],
                                locator=f"Table1;{year};{match[1]};{metric}",
                                definition="malaria positive by microscopy/RDT",
                                quality="arithmetic_review" if micro + rdt != cases else "observed",
                                note="2020 March cases 476 differ from microscopy 90 plus RDT 336; published case totals match monthly case sums; no silent repair",
                            )
                        )
        elif "/bik/" in item["url"]:
            names = [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]
            table = text.split("TABLE 1. Malaria incidence distribution by months in 2019")[
                1
            ].split("Total")[0]
            for match in re.finditer(
                r"^\s*([A-Za-z]+)\s+(\d+)\s+(\d+)\s+\([\d.]+\)\s+(\d+)\s+\([\d.]+\)", table, re.M
            ):
                if match[1] not in names:
                    continue
                total, positive, negative = map(int, match.group(2, 3, 4))
                assert total == positive + negative
                for metric, value in [
                    ("cases", positive),
                    ("blood_samples", total),
                    ("negative_tests", negative),
                ]:
                    rows.append(
                        observation(
                            "weoe_paper",
                            "5321:puskesmas:Weoe",
                            "Weoe",
                            "facility",
                            f"2019-{names.index(match[1]) + 1:02}",
                            value,
                            disease="malaria",
                            metric=metric,
                            url=item["url"],
                            path=item["path"],
                            locator=f"Table1;{match[1]};{metric}",
                            definition="microscopy-confirmed malaria in the eligible study sample",
                            quality="study_subset",
                            note="Study includes 815 of 860 laboratory tests; excludes 44 outside catchment and one incomplete record. Not a complete facility surveillance count.",
                        )
                    )
    (ROOT / "paper_observations.json").write_text(json.dumps(rows, indent=2))
    print("Extracted", len(rows))

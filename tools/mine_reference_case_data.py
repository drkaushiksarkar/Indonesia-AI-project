"""Fetch pinned provincial dengue counts and the public Yogyakarta case workbook."""

import hashlib
import json
from pathlib import Path

import rdata
import requests

ROOT = Path("../outputs/extensive_mining")
COMMIT = "c7dd60c9a0547723a6f0b156cee1a86b5604fbeb"


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    path.write_bytes(r.content)
    return {
        "url": url,
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(r.content).hexdigest(),
        "status": "available",
    }


if __name__ == "__main__":
    p = ROOT / "research/dengue_data_admin1_indonesia.rds"
    items = [
        fetch(
            f"https://raw.githubusercontent.com/mlgh-sg/dengue-climate-indonesia/{COMMIT}/data/{p.name}",
            p,
        )
    ]
    frame = rdata.read_rds(p)
    frame.to_csv(p.with_suffix(".csv"), index=False)
    metadata = requests.get("https://api.figshare.com/v2/articles/12199688", timeout=45)
    metadata.raise_for_status()
    (ROOT / "primary/figshare12199688.json").write_text(json.dumps(metadata.json(), indent=2))
    for item in metadata.json()["files"]:
        if item["name"] == "Figure 5_6_DengueCaseData.xlsx":
            items.append(fetch(item["download_url"], ROOT / "primary" / item["name"]))
    (ROOT / "reference_cases_manifest.json").write_text(json.dumps(items, indent=2))
    print("Case references", len(items), "provincial rows", len(frame))

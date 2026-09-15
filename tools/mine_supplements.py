"""Cache publicly linked malaria aggregate workbooks and Bandung research supplement."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

OUT = Path("../outputs/extensive_mining/supplements")
ITEMS = [
    (
        f"pmed.1002815.s00{n}.xlsx",
        f"https://journals.plos.org/plosmedicine/article/file?type=supplementary&id=10.1371/journal.pmed.1002815.s00{n}",
    )
    for n in [6, 7, 8]
] + [
    (
        "pmed.1002815.s004.docx",
        "https://journals.plos.org/plosmedicine/article/file?type=supplementary&id=10.1371/journal.pmed.1002815.s004",
    ),
    (
        "bandung_supplement.zip",
        "https://static-content.springer-cdn.com/esm/art%3A10.1007%2Fs10109-021-00368-0/MediaObjects/10109_2021_368_MOESM2_ESM.zip",
    ),
    (
        "bandung_supplement.pdf",
        "https://static-content.springer-cdn.com/esm/art%3A10.1007%2Fs10109-021-00368-0/MediaObjects/10109_2021_368_MOESM1_ESM.pdf",
    ),
]


def fetch(item):
    name, url = item
    r = {"name": name, "url": url}
    try:
        response = requests.get(url, timeout=(10, 60))
        response.raise_for_status()
        data = response.content
        if not data.startswith((b"PK", b"%PDF")):
            raise ValueError("Unexpected non-document response")
        path = OUT / name
        path.write_bytes(data)
        r.update(
            status="available",
            path=str(path.resolve()),
            bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )
    except Exception as e:
        r.update(status="failed", error=str(e))
    return r


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, ITEMS))
    (OUT / "manifest.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

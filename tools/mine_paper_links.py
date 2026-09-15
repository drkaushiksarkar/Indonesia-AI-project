"""Cache public disease research pages and list linked supplementary datasets."""

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

OUT = Path("../outputs/extensive_mining/literature")
SOURCES = [
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC6541239/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12135496/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11881261/",
    "https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2026.1872970/full",
    "https://www.tycho.pitt.edu/dataset/ID.38362002/",
    "https://data.mendeley.com/datasets/x855pphhx9/1",
    "https://e-arsip.bontangkota.go.id/images/Aplikasi_DBD.pdf",
    "https://repository.badankebijakan.kemkes.go.id/4900/1/Lap_Vektora_SUMSEL_2015%20%281%29.pdf",
    "https://repository.badankebijakan.kemkes.go.id/4902/1/Lap_Vektora_PAPUA_2015%20%281%29.pdf",
]


def fetch(url):
    d = {"url": url}
    key = hashlib.sha256(url.encode()).hexdigest()[:18]
    try:
        r = requests.get(url, timeout=(10, 45))
        r.raise_for_status()
        ispdf = r.content.startswith(b"%PDF")
        path = OUT / (key + (".pdf" if ispdf else ".html"))
        path.write_bytes(r.content)
        if ispdf:
            subprocess.run(
                ["pdftotext", "-layout", str(path), str(path.with_suffix(".txt"))],
                check=True,
                timeout=45,
                capture_output=True,
            )
        soup = BeautifulSoup(r.text, "html.parser") if not ispdf else None
        d.update(
            status="available",
            path=str(path.resolve()),
            sha256=hashlib.sha256(r.content).hexdigest(),
            links=[
                {"label": a.get_text(" ", strip=True), "url": urljoin(url, a["href"])}
                for a in soup.select("a[href]")
                if any(
                    t in (a["href"] + " " + a.get_text()).lower()
                    for t in [
                        "supplement",
                        "figshare",
                        "zenodo",
                        "dryad",
                        "appendix",
                        "additional file",
                        "s1 data",
                        "s2 data",
                        "download",
                        "mendeley",
                    ]
                )
            ]
            if soup
            else [],
        )
    except Exception as e:
        d.update(status="failed", error=str(e))
    return d


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, SOURCES))
    (OUT / "manifest.json").write_text(json.dumps(results, indent=2))
    for r in results:
        print(r["url"], r["status"], r.get("links", []), flush=True)

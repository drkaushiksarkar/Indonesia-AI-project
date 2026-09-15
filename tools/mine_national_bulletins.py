"""Cache publicly linked Ministry disease bulletins for case-table extraction."""

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

ROOT = Path("../outputs/extensive_mining/national_bulletins")
URL = "https://surveilans.kemkes.go.id/publikasi"


def fetch(url):
    path = ROOT / (hashlib.sha256(url.encode()).hexdigest()[:18] + ".pdf")
    item = {"url": url}
    try:
        if not path.exists():
            r = requests.get(url, timeout=(10, 45))
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                raise ValueError("Non-PDF response")
            path.write_bytes(r.content)
        text = path.with_suffix(".txt")
        subprocess.run(
            ["pdftotext", "-layout", str(path), str(text)],
            check=True,
            timeout=45,
            capture_output=True,
        )
        item.update(
            status="available",
            path=str(path.resolve()),
            text_path=str(text.resolve()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    except Exception as e:
        item.update(status="failed", error=str(e))
    return item


if __name__ == "__main__":
    ROOT.mkdir(exist_ok=True)
    r = requests.get(URL, timeout=40)
    r.raise_for_status()
    (ROOT / "archive.html").write_text(r.text)
    soup = BeautifulSoup(r.text, "html.parser")
    urls = sorted(
        {
            urljoin(URL, a["href"])
            for a in soup.select("a[href]")
            if a["href"].lower().endswith(".pdf")
            and any(
                w in a["href"].lower()
                for w in ["laporan-mingguan", "weekly-report", "penyakit-vektor", "potensial-klb"]
            )
        }
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(fetch, urls))
    (ROOT / "manifest.json").write_text(json.dumps(result, indent=2))
    print("Bulletins", len(result), "available", sum(x["status"] == "available" for x in result))

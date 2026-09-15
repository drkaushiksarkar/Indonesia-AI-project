"""Download PDF file targets embedded by the public official bulletin viewer."""

import hashlib
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup

ROOT = Path("../outputs/extensive_mining/inhu")


def fetch(item):
    url, landing = item
    r = {"url": url, "landing_path": str(landing)}
    key = hashlib.sha256(url.encode()).hexdigest()[:18]
    path = ROOT / "pdf" / f"{key}.pdf"
    try:
        if not path.exists():
            response = requests.get(url, timeout=(10, 45))
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                raise ValueError("Not a PDF")
            path.write_bytes(response.content)
        textpath = path.with_suffix(".txt")
        if not textpath.exists():
            subprocess.run(
                ["pdftotext", "-layout", str(path), str(textpath)], check=True, timeout=40
            )
        r.update(
            status="available",
            path=str(path.resolve()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            text_path=str(textpath.resolve()),
        )
    except Exception as e:
        r.update(status="failed", error=str(e))
    return r


if __name__ == "__main__":
    (ROOT / "pdf").mkdir(exist_ok=True)
    urls = {}
    for f in ROOT.glob("*.html"):
        soup = BeautifulSoup(f.read_text(), "html.parser")
        for a in soup.select("iframe[src],a[href]"):
            u = a.get("src") or a.get("href")
            target = parse_qs(urlparse(u).query).get("file", [u])[0]
            drive = re.search(r"https://drive.google.com/file/d/([A-Za-z0-9_-]+)/", target)
            if drive:
                target = f"https://drive.google.com/uc?export=download&id={drive[1]}"
                urls[target] = f
            if target.startswith("https://dinkes.inhukab.go.id/") and ".pdf" in target:
                urls[target] = f
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, urls.items()))
    (ROOT / "pdf_manifest.json").write_text(json.dumps(results, indent=2))
    print("PDFs", len(results), "available", sum(x["status"] == "available" for x in results))

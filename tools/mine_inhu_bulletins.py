"""Follow official Indragiri Hulu weekly bulletin archives and public document links."""

import concurrent.futures
import hashlib
import json
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

ROOT = Path("../outputs/extensive_mining/inhu")
BASE = "https://dinkes.inhukab.go.id"


def get(url):
    r = requests.get(url, timeout=(10, 40))
    r.raise_for_status()
    return r


def page(url):
    out = {"url": url}
    try:
        path = ROOT / (hashlib.sha256(url.encode()).hexdigest()[:18] + ".html")
        if not path.exists():
            r = get(url)
            path.write_text(r.text)
        soup = BeautifulSoup(path.read_text(), "html.parser")
        links = []
        for a in soup.select("a[href],iframe[src],embed[src],object[data]"):
            href = a.get("href") or a.get("src") or a.get("data")
            u = urljoin(url, href)
            if any(t in u for t in [".pdf", "drive.google.com", "download=", "wpdmdl="]):
                links.append(u)
        out.update(status="available", path=str(path.resolve()), links=sorted(set(links)))
    except Exception as e:
        out.update(status="failed", error=str(e))
    return out


if __name__ == "__main__":
    ROOT.mkdir(exist_ok=True)
    urls = set()
    for year in [2024, 2025, 2026]:
        url = f"{BASE}/buletin-skdr-tahun-{year}/"
        try:
            r = get(url)
            (ROOT / f"archive{year}.html").write_text(r.text)
            s = BeautifulSoup(r.text, "html.parser")
            urls |= {
                urljoin(url, a["href"])
                for a in s.select("a[href]")
                if "buletin-skdr-minggu-" in a["href"]
            }
        except Exception as e:
            print(year, str(e), flush=True)
    print("Bulletins", len(urls), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(page, sorted(urls)))
    (ROOT / "manifest.json").write_text(json.dumps(results, indent=2))
    print(
        "Pages",
        len(results),
        "documents",
        sum(len(x.get("links", [])) for x in results),
        flush=True,
    )

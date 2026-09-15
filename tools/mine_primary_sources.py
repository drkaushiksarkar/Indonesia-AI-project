import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from bs4 import BeautifulSoup

OUT = Path("../outputs/extensive_mining/primary")
OUT.mkdir(exist_ok=True)
URLS = {
    "semarang_dashboard": "https://lekminkes.dinkes.semarangkota.go.id/",
    "yogyakarta_prediction": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0152688",
    "yogyakarta_wolbachia": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7403856/",
    "bandung_knmi": "https://cdn.knmi.nl/knmi/pdf/bibliotheek/knmipubIR/IR2013-06.pdf",
    "semarang2023": "https://pustakadata.semarangkota.go.id/upload/pdf/463-buku-profil-kesehatan-tahun-2023.pdf",
}


def get(pair):
    name, url = pair
    try:
        r = requests.get(url, timeout=(10, 60))
        r.raise_for_status()
        suffix = ".pdf" if r.content.startswith(b"%PDF") else ".html"
        (OUT / (name + suffix)).write_bytes(r.content)
        result = {"name": name, "url": url, "size": len(r.content), "status": "available"}
        if suffix == ".html":
            soup = BeautifulSoup(r.text, "html.parser")
            result["links"] = [
                {"text": a.get_text(" ", strip=True), "url": a.get("href")}
                for a in soup.find_all("a", href=True)
                if any(
                    k in (a.get("href", "") + " " + a.get_text()).lower()
                    for k in ["supp", "support", "dataset", "xlsx", "csv", "s1", "download"]
                )
            ]
        print(name, len(r.content), flush=True)
    except Exception as e:
        result = {"name": name, "url": url, "status": "failed", "error": str(e)}
    return result


with ThreadPoolExecutor(max_workers=5) as pool:
    rows = list(pool.map(get, URLS.items()))
(OUT / "manifest.json").write_text(json.dumps(rows, indent=2))

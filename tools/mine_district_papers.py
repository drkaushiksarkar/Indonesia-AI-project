"""Cache public district dengue papers with explicit monthly case tables."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mine_paper_links import fetch

URLS = [
    "https://ojs.serambimekkah.ac.id/jse/article/download/2650/2116",
    "https://ejurnalmalahayati.ac.id/index.php/medika/article/viewFile/17967/pdf",
]
if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(fetch, URLS))
    for item in results:
        if item["status"] == "available" and not item["path"].endswith(".pdf"):
            item.update(
                status="failed", error="Expected public PDF; received HTML landing/error page"
            )
    Path("../outputs/extensive_mining/literature/district_papers_manifest.json").write_text(
        json.dumps(results, indent=2)
    )

"""Read public Europe PMC full-text XML and cache linked disease supplements."""

import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path("../outputs/extensive_mining/literature")
IDS = ["PMC6541239", "PMC12135496", "PMC11881261", "PMC8857957", "PMC11628951"]


def fetch(pmc):
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML"
    try:
        r = requests.get(url, timeout=40)
        r.raise_for_status()
        (ROOT / f"{pmc}.xml").write_text(r.text)
        soup = ET.fromstring(r.content)
        print(
            pmc,
            [
                (ET.tostring(x, encoding="unicode")[:300])
                for x in soup.iter()
                if x.tag in ["supplementary-material", "ext-link"]
                if any(
                    t in ET.tostring(x, encoding="unicode").lower()
                    for t in ["data", "supplement", "figshare", "doi.org", "github"]
                )
            ][-20:],
            flush=True,
        )
    except Exception as e:
        print(pmc, str(e), flush=True)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, IDS))

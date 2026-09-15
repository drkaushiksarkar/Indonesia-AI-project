"""Search public CKAN origins observed in existing national-catalog resource URLs."""

import concurrent.futures
import json
from pathlib import Path
from urllib.parse import urlparse

from mine_catalogs import mine

ROOT = Path("../outputs/extensive_mining")
if __name__ == "__main__":
    done = set()
    for p in (ROOT / "catalogs").glob("*.json"):
        d = json.loads(p.read_text())
        if isinstance(d, dict):
            done.add(d["catalog"])
    origins = json.loads((ROOT / "observed_ckan_origins.json").read_text())
    origins = [o for o in origins if o not in done and urlparse(o).hostname.endswith(".go.id")]
    print("Additional observed public portals", len(origins), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(mine, origins))

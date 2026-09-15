"""Retry incomplete query pages without discarding successfully cached catalogs."""

import json
from pathlib import Path

import requests

ROOT = Path("../outputs/extensive_mining/catalogs")
if __name__ == "__main__":
    for path in ROOT.glob("*.json"):
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or not data.get("packages"):
            continue
        packages = {p["id"]: p for p in data["packages"]}
        for receipt in data["receipts"]:
            if receipt["status"] == "complete":
                continue
            start = receipt["retrieved"]
            receipt["previous_error"] = receipt.get("error")
            try:
                while True:
                    r = requests.get(
                        data["catalog"] + "/api/3/action/package_search",
                        params={"q": receipt["query"], "start": start, "rows": 50},
                        timeout=(10, 40),
                    )
                    r.raise_for_status()
                    result = r.json()["result"]
                    batch = result["results"]
                    packages.update({p["id"]: p for p in batch})
                    start += len(batch)
                    receipt["retrieved"] = start
                    if not batch or start >= result["count"]:
                        break
                receipt["status"] = "complete_after_retry"
                receipt.pop("error", None)
            except Exception as e:
                receipt["error"] = str(e)
            print(data["catalog"], receipt["query"], start, receipt["status"], flush=True)
        data["packages"] = list(packages.values())
        path.write_text(json.dumps(data, ensure_ascii=False))

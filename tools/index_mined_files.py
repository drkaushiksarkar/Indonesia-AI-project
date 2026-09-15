"""Build local text/table evidence for downloaded public aggregate resources."""

import concurrent.futures
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

OUT = Path("../outputs/extensive_mining")


def inspect(item):
    result = {k: item.get(k) for k in ["resource_id", "title", "path", "url", "format"]}
    try:
        path = Path(item["path"])
        target = OUT / "text" / f"{item['resource_id']}.txt"
        if target.exists():
            text = target.read_text()
        elif item["format"] == "pdf":
            text = subprocess.check_output(
                ["pdftotext", "-layout", str(path), "-"], timeout=40
            ).decode(errors="replace")
        elif item["format"] in ["xlsx", "xls"]:
            sheets = pd.read_excel(path, sheet_name=None, header=None)
            text = "\n".join(
                "SHEET " + str(n) + "\n" + d.fillna("").to_csv(index=False, header=False)
                for n, d in sheets.items()
            )
        else:
            text = path.read_text(errors="replace")
        if re.search(r"(?im)^.*\bnama\b.*\bumur\b.*\balamat\b", text):
            result.update(
                status="excluded_personal_health_register", chars=len(text), month_labels=0
            )
            target.write_text(
                "[Excluded: individual health register. Do not use or redistribute.]\n"
            )
            return result
        target.write_text(text)
        months = set(
            re.findall(
                r"\b(?:januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember|jan|feb|mar|apr|jun|jul|agu|agt|sep|okt|nov|des)\b",
                text.lower(),
            )
        )
        result.update(
            text_path=str(target.resolve()),
            chars=len(text),
            month_labels=len(months),
            status="extracted",
        )
    except Exception as e:
        result.update(status="failed", error=str(e))
    return result


if __name__ == "__main__":
    (OUT / "text").mkdir(exist_ok=True)
    manifest = json.loads((OUT / "download_manifest.json").read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(inspect, [x for x in manifest if x["status"] == "available"]))
    (OUT / "text_manifest.json").write_text(json.dumps(result, indent=2))
    print("Extracted", sum(x["status"] == "extracted" for x in result), "of", len(result))
    for x in result:
        if x.get("month_labels", 0) >= 6:
            print(x["resource_id"], x["title"], x["month_labels"])

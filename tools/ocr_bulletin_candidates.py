"""Create OCR evidence for image-based charts; do not auto-promote OCR as case data."""

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pdfplumber

ROOT = Path("../outputs/extensive_mining/national_bulletins")


def extract(item):
    key, source, page = item
    destination = ROOT / "ocr" / key
    with pdfplumber.open(source["path"]) as pdf:
        p = pdf.pages[page - 1]
        p.crop((0, 0, p.width * 0.47, p.height)).to_image(resolution=350).original.save(
            destination.with_suffix(".png")
        )
    result = subprocess.run(
        ["tesseract", str(destination.with_suffix(".png")), "stdout", "--psm", "6"],
        capture_output=True,
        text=True,
        check=True,
    )
    destination.with_suffix(".txt").write_text(result.stdout)
    return {
        "source_url": source["url"],
        "page": page,
        "text_path": str(destination.with_suffix(".txt").resolve()),
        "image_path": str(destination.with_suffix(".png").resolve()),
        "status": "ocr_requires_visual_verification",
    }


if __name__ == "__main__":
    (ROOT / "ocr").mkdir(exist_ok=True)
    manifest = {x["url"]: x for x in json.loads((ROOT / "manifest.json").read_text())}
    work = {}
    for item in json.loads((ROOT / "extraction_audit.json").read_text()):
        if item.get("rows", 0) == 38:
            continue
        source = manifest[item["url"]]
        key = hashlib.sha256((source["sha256"] + str(item["page"])).encode()).hexdigest()[:18]
        work[key] = (key, source, item["page"])
    with ThreadPoolExecutor(max_workers=1) as pool:
        results = list(pool.map(extract, work.values()))
    (ROOT / "ocr_manifest.json").write_text(json.dumps(results, indent=2))
    print("OCR review candidates", len(results))

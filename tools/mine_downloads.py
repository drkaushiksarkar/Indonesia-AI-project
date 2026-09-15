"""Download candidate public aggregate disease resources; retain a provenance manifest."""

import concurrent.futures
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import psycopg
import requests

OUT = Path("../outputs/extensive_mining")
RAW = OUT / "raw"
RAW.mkdir(exist_ok=True)


def download(item):
    item = item.copy()
    path = Path(
        item.get("cached_path")
        or RAW / (hashlib.sha256(item["url"].encode()).hexdigest()[:24] + "." + item["format"])
    )
    try:
        if not path.is_file() or path.stat().st_size == 0:
            response = requests.get(item["url"], timeout=(10, 45), stream=True)
            response.raise_for_status()
            if int(response.headers.get("Content-Length", 0)) > 100_000_000:
                raise ValueError("Resource over 100MB; separately assess")
            data = bytearray()
            for chunk in response.iter_content(1024 * 1024):
                data.extend(chunk)
                if len(data) > 100_000_000:
                    raise ValueError("Resource over 100MB")
            if not data or bytes(data[:100]).lstrip().lower().startswith(
                (b"<!doctype html", b"<html")
            ):
                raise ValueError("Empty resource or HTML instead of data")
            path.write_bytes(data)
        item.update(
            path=str(path.resolve()),
            size=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            status="available",
        )
    except Exception as exc:
        item.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return item


if __name__ == "__main__":
    with psycopg.connect(dbname="indonesia_vector_lake") as conn:
        cached = {
            p["resource_id"]: p.get("path", "")
            for (p,) in conn.execute("select payload from bronze.record where dataset_id=39")
        }
    candidates = {}
    for f in (OUT / "catalogs").glob("*.json"):
        d = json.loads(f.read_text())
        if not isinstance(d, dict):
            continue
        for p in d.get("packages", []):
            title = p.get("title", "")
            if not re.search(r"\b(dbd|dengue|malaria)\b|demam berdarah", title, re.I):
                continue
            if re.search(r"dummy|sintetis|synthetic", p.get("notes", ""), re.I):
                continue
            for r in p.get("resources", []):
                url = r.get("url", "")
                suffix = Path(urlparse(url).path).suffix.lower().lstrip(".")
                fmt = (
                    suffix
                    if suffix in {"pdf", "csv", "xls", "xlsx", "json", "zip"}
                    else str(r.get("format", "")).lower()
                )
                if fmt not in {"pdf", "csv", "xls", "xlsx", "json", "zip"} or not url.startswith(
                    ("http://", "https://")
                ):
                    continue
                candidates[url] = {
                    "catalog": d["catalog"],
                    "dataset_id": p["id"],
                    "title": title,
                    "organization": (p.get("organization") or {}).get("title", ""),
                    "resource_id": r["id"],
                    "resource_name": r.get("name", ""),
                    "url": url,
                    "format": fmt,
                    "cached_path": cached.get(r["id"], ""),
                    "metadata_modified": p.get("metadata_modified"),
                }
    print("Resources", len(candidates), flush=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for result in pool.map(download, candidates.values()):
            results.append(result)
            if len(results) % 25 == 0:
                print(
                    "Processed",
                    len(results),
                    "available",
                    sum(r["status"] == "available" for r in results),
                    flush=True,
                )
                (OUT / "download_manifest.json").write_text(json.dumps(results, ensure_ascii=False))
    (OUT / "download_manifest.json").write_text(json.dumps(results, ensure_ascii=False))
    print(
        "Done",
        len(results),
        "available",
        sum(r["status"] == "available" for r in results),
        flush=True,
    )

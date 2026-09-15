from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

BLUE = "#276FBF"
GOLD = "#D39A22"
GREEN = "#3D8B64"
GREY = "#D9E0E7"
INK = "#1F2933"
WHITE = "#FFFFFF"


def _bars(
    draw: ImageDraw.ImageDraw,
    values: pd.Series,
    bounds: tuple[int, int, int, int],
    title: str,
    color: str,
    font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = bounds
    draw.text((left, top), title, fill=INK, font=font)
    values = values.sort_values(ascending=False).head(12).sort_values()
    if values.empty:
        return
    maximum = max(float(values.max()), 1.0)
    height = max(18, (bottom - top - 55) // len(values))
    label_width = 210
    for index, (label, value) in enumerate(values.items()):
        y = bottom - (index + 1) * height
        width = int((right - left - label_width - 55) * float(value) / maximum)
        draw.text((left, y + 2), str(label)[:28], fill=INK, font=font)
        draw.rectangle(
            (left + label_width, y, left + label_width + width, y + height - 5),
            fill=color,
        )
        draw.text((left + label_width + width + 6, y + 2), str(int(value)), fill=INK, font=font)


def create_public_harvest_figure(
    output: Path,
    resources: pd.DataFrame,
    catalog: pd.DataFrame,
    dashboards: pd.DataFrame,
) -> list[Path]:
    if resources.empty and catalog.empty and dashboards.empty:
        return []
    output.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1600, 760), WHITE)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((50, 28), "Public dengue and malaria source harvest", fill=INK, font=font)
    resource_counts = (
        resources.groupby("source").size() if not resources.empty else pd.Series(dtype="int64")
    )
    _bars(draw, resource_counts, (50, 80, 770, 700), "Resources by source", BLUE, font)
    components = []
    if not catalog.empty and "disease" in catalog:
        components.append(catalog.groupby("disease").size())
    if not dashboards.empty:
        components.append(dashboards.groupby("status").size())
    component_counts = (
        pd.concat(components).groupby(level=0).sum() if components else pd.Series(dtype="int64")
    )
    _bars(
        draw,
        component_counts,
        (830, 80, 1550, 700),
        "Datasets and dashboard calls",
        GOLD,
        font,
    )
    draw.line((50, 720, 1550, 720), fill=GREY, width=2)
    downloaded = (
        int(resources["status"].isin(["cached", "downloaded"]).sum()) if not resources.empty else 0
    )
    draw.text(
        (50, 730),
        f"{len(catalog)} datasets | {len(resources)} resources | {downloaded} retrieved",
        fill=GREEN,
        font=font,
    )
    path = output / "public_source_harvest.png"
    image.save(path)
    return [path]

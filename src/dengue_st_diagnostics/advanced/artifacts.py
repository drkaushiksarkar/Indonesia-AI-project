from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as PlotPolygon
from PIL import Image

COLORS = {
    "observed": "#152238",
    "trend": "#007C91",
    "seasonal": "#F28E2B",
    "residual": "#8C6BB1",
    "anomaly": "#D62728",
    "cold": "#2C7FB8",
    "hot": "#D7301F",
}


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def temporal_artifacts(
    root: Path,
    series: pd.Series,
    plot_data: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    series_id = str(series["series_id"])
    title = f"{series['disease'] or 'Unspecified'} · {series['metric']} · {series['location_name']}"
    if "stl" in plot_data:
        value = plot_data["stl"]
        figure, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
        for axis, name, color in zip(
            axes,
            ("observed", "trend", "seasonal", "residual"),
            (COLORS["observed"], COLORS["trend"], COLORS["seasonal"], COLORS["residual"]),
            strict=True,
        ):
            axis.plot(value["dates"], value[name], color=color, linewidth=1.5)
            axis.set_ylabel(name.title())
            axis.grid(alpha=0.2)
        figure.suptitle(f"Robust STL decomposition\n{title}")
        path = root / "stl" / f"{series_id}.png"
        _save(figure, path)
        records.append(
            {
                "path": path,
                "method_id": "stl",
                "series_id": series_id,
                "panel_id": series["panel_id"],
                "artifact_role": "stl_decomposition",
            }
        )
    if "anomaly" in plot_data:
        value = plot_data["anomaly"]
        figure, axes = plt.subplots(
            2, 1, figsize=(13, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )
        axes[0].plot(value["dates"], value["observed"], color=COLORS["observed"], linewidth=1.5)
        anomaly_dates = {
            event["period_start"]
            for event in value["events"]
            if event["method_id"] in {"robust_residual_anomaly", "robust_multivariate_anomaly"}
        }
        mask = np.array([date.date() in anomaly_dates for date in value["dates"]])
        axes[0].scatter(
            np.asarray(value["dates"])[mask],
            np.asarray(value["observed"])[mask],
            color=COLORS["anomaly"],
            s=38,
            zorder=3,
        )
        axes[0].set_ylabel("Observed value")
        axes[0].grid(alpha=0.2)
        axes[1].plot(value["dates"], value["scores"], color=COLORS["residual"], linewidth=1.2)
        axes[1].axhline(3.5, color=COLORS["anomaly"], linestyle="--", linewidth=1)
        axes[1].axhline(-3.5, color=COLORS["anomaly"], linestyle="--", linewidth=1)
        axes[1].set_ylabel("Robust score")
        axes[1].grid(alpha=0.2)
        figure.suptitle(f"Anomaly diagnostics\n{title}")
        path = root / "anomaly" / f"{series_id}.png"
        _save(figure, path)
        records.append(
            {
                "path": path,
                "method_id": "robust_residual_anomaly",
                "series_id": series_id,
                "panel_id": series["panel_id"],
                "artifact_role": "anomaly_diagnostic",
            }
        )
    if "wavelet" in plot_data:
        value = plot_data["wavelet"]
        figure, axis = plt.subplots(figsize=(13, 5))
        image = axis.pcolormesh(
            value["dates"], value["scales"], value["power"], shading="auto", cmap="magma"
        )
        axis.invert_yaxis()
        axis.set_ylabel("Wavelet scale")
        axis.set_title(f"Continuous wavelet power\n{title}")
        figure.colorbar(image, ax=axis, label="Power")
        path = root / "wavelet" / f"{series_id}.png"
        _save(figure, path)
        records.append(
            {
                "path": path,
                "method_id": "wavelet",
                "series_id": series_id,
                "panel_id": series["panel_id"],
                "artifact_role": "wavelet_power",
            }
        )
    if "emd" in plot_data:
        value = plot_data["emd"]
        count = len(value["imfs"]) + 2
        figure, axes = plt.subplots(count, 1, figsize=(13, max(7, 1.7 * count)), sharex=True)
        axes[0].plot(value["dates"], value["observed"], color=COLORS["observed"], linewidth=1.2)
        axes[0].set_ylabel("Observed")
        for index, imf in enumerate(value["imfs"], start=1):
            axes[index].plot(
                value["dates"], imf, color=plt.cm.viridis(index / max(1, len(value["imfs"])))
            )
            axes[index].set_ylabel(f"IMF {index}")
        axes[-1].plot(value["dates"], value["residue"], color=COLORS["trend"])
        axes[-1].set_ylabel("Residue")
        for axis in axes:
            axis.grid(alpha=0.2)
        figure.suptitle(f"Empirical mode decomposition\n{title}")
        path = root / "emd_hht" / f"{series_id}.png"
        _save(figure, path)
        records.append(
            {
                "path": path,
                "method_id": "emd_hht",
                "series_id": series_id,
                "panel_id": series["panel_id"],
                "artifact_role": "imf_decomposition",
            }
        )
    return records


def panel_artifacts(
    root: Path,
    panel_id: str,
    title: str,
    plot_data: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if "pca" in plot_data:
        value = plot_data["pca"]
        scores = np.asarray(value["scores"])
        figure, axis = plt.subplots(figsize=(10, 8))
        axis.scatter(scores[:, 0], scores[:, 1], color="#007C91", alpha=0.75, s=34)
        distance = np.linalg.norm(scores, axis=1)
        selected = np.argsort(distance)[-min(10, len(distance)) :]
        selected = selected[np.argsort(scores[selected, 1])]
        for position, index in enumerate(selected):
            label_parts = [part.strip() for part in str(value["labels"][index]).split("|")]
            label = " | ".join(label_parts[-2:])
            horizontal = 6 if scores[index, 0] >= 0 else -6
            vertical = 4 + 8 * (position % 3)
            axis.annotate(
                label,
                scores[index],
                fontsize=7,
                xytext=(horizontal, vertical),
                textcoords="offset points",
                ha="left" if horizontal > 0 else "right",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1},
            )
        loadings = np.asarray(value["loadings"])
        if loadings.shape[1] > 0:
            magnitude = np.linalg.norm(loadings, axis=0)
            loading_indices = np.argsort(magnitude)[-min(8, len(magnitude)) :]
            scale = max(np.ptp(scores[:, 0]), np.ptp(scores[:, 1]), 1)
            for index in loading_indices:
                x_value = loadings[0, index] * scale
                y_value = loadings[1, index] * scale
                axis.arrow(
                    0,
                    0,
                    x_value,
                    y_value,
                    color="#F28E2B",
                    alpha=0.65,
                    width=0.003 * scale,
                    length_includes_head=True,
                )
                axis.text(
                    x_value,
                    y_value,
                    pd.Timestamp(value["periods"][index]).strftime("%Y-%m"),
                    fontsize=6,
                    color="#8A4F00",
                )
        variance = value["variance"]
        axis.set_xlabel(f"PC1 ({100 * variance[0]:.1f}% variance)")
        axis.set_ylabel(f"PC2 ({100 * variance[1]:.1f}% variance)")
        axis.axhline(0, color="#B0B0B0", linewidth=0.7)
        axis.axvline(0, color="#B0B0B0", linewidth=0.7)
        axis.grid(alpha=0.2)
        axis.set_title(f"High-dimensional geography PCA biplot\n{title}")
        path = root / "pca_biplot" / f"{panel_id}.png"
        _save(figure, path)
        records.append(
            {
                "path": path,
                "method_id": "pca_biplot",
                "series_id": None,
                "panel_id": panel_id,
                "artifact_role": "pca_biplot",
            }
        )
    if "correlation" in plot_data:
        value = plot_data["correlation"]
        matrix = np.asarray(value["matrix"])
        labels = list(value["labels"])
        if len(matrix) > 80:
            importance = np.nanmean(np.abs(matrix - np.eye(len(matrix))), axis=1)
            selected = np.argsort(importance)[-80:]
            matrix = matrix[np.ix_(selected, selected)]
            labels = [labels[index] for index in selected]
        figure, axis = plt.subplots(figsize=(11, 9))
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        if len(labels) <= 35:
            axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=90, fontsize=6)
            axis.set_yticks(np.arange(len(labels)), labels=labels, fontsize=6)
        else:
            axis.set_xticks([])
            axis.set_yticks([])
        axis.set_title(f"Geographic temporal synchrony\n{title}")
        figure.colorbar(image, ax=axis, label="Pearson correlation")
        path = root / "synchrony" / f"{panel_id}.png"
        _save(figure, path)
        records.append(
            {
                "path": path,
                "method_id": "panel_synchrony",
                "series_id": None,
                "panel_id": panel_id,
                "artifact_role": "synchrony_matrix",
            }
        )
    if "dmd" in plot_data:
        value = plot_data["dmd"]
        eigenvalues = np.asarray(value["eigenvalues"])
        amplitudes = np.asarray(value["amplitudes"])
        figure, axis = plt.subplots(figsize=(8, 8))
        angles = np.linspace(0, 2 * np.pi, 400)
        axis.plot(np.cos(angles), np.sin(angles), color="#888888", linestyle="--", linewidth=1)
        sizes = 30 + 170 * amplitudes / max(np.max(amplitudes), 1e-12)
        axis.scatter(
            np.real(eigenvalues),
            np.imag(eigenvalues),
            s=sizes,
            c=np.angle(eigenvalues),
            cmap="twilight",
            alpha=0.8,
        )
        axis.axhline(0, color="#B0B0B0", linewidth=0.7)
        axis.axvline(0, color="#B0B0B0", linewidth=0.7)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Real eigenvalue")
        axis.set_ylabel("Imaginary eigenvalue")
        axis.set_title(f"Dynamic mode spectrum\n{title}")
        path = root / "dynamic_modes" / f"{panel_id}.png"
        _save(figure, path)
        records.append(
            {
                "path": path,
                "method_id": "dynamic_mode_decomposition",
                "series_id": None,
                "panel_id": panel_id,
                "artifact_role": "dynamic_mode_spectrum",
            }
        )
    return records


def _polygon_patches(geometry: Any) -> list[PlotPolygon]:
    polygons = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
    return [PlotPolygon(np.asarray(polygon.exterior.coords), closed=True) for polygon in polygons]


def spatial_artifact(
    root: Path,
    panel_id: str,
    title: str,
    plot_data: dict[str, Any],
) -> list[dict[str, Any]]:
    if "spatial" not in plot_data:
        return []
    value = plot_data["spatial"]
    figure, axes = plt.subplots(1, 3, figsize=(18, 7))
    fields = [
        ("values", "Observed value", "viridis", None, None),
        ("local_i", "Local Moran I", "coolwarm", None, None),
        ("hotspot_z", "Getis-Ord Gi* z", "coolwarm", -3, 3),
    ]
    for axis, (field, label, cmap, minimum, maximum) in zip(axes, fields, strict=True):
        patches: list[PlotPolygon] = []
        colors: list[float] = []
        for record, field_value in zip(value["records"], value[field], strict=True):
            current = _polygon_patches(record["geometry"])
            patches.extend(current)
            colors.extend([field_value] * len(current))
        collection = PatchCollection(patches, cmap=cmap, edgecolor="white", linewidth=0.35)
        collection.set_array(np.asarray(colors))
        if minimum is not None and maximum is not None:
            collection.set_clim(minimum, maximum)
        axis.add_collection(collection)
        axis.autoscale_view()
        axis.set_aspect("equal", adjustable="box")
        axis.axis("off")
        axis.set_title(label)
        figure.colorbar(collection, ax=axis, fraction=0.035, pad=0.02)
    period = pd.Timestamp(value["period"]).strftime("%B %Y")
    figure.suptitle(f"Spatial diagnostics · {period}\n{title}")
    path = root / "spatial" / f"{panel_id}.png"
    _save(figure, path)
    return [
        {
            "path": path,
            "method_id": "moran_local",
            "series_id": None,
            "panel_id": panel_id,
            "artifact_role": "spatial_diagnostic_map",
        }
    ]


def overview_artifacts(
    root: Path,
    catalog: pd.DataFrame,
    statistics: pd.DataFrame,
    events: pd.DataFrame,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    grain = catalog.groupby(["spatial_resolution", "temporal_resolution"], as_index=False).agg(
        series=("series_id", "nunique"), observations=("observations", "sum")
    )
    pivot = grain.pivot(
        index="spatial_resolution", columns="temporal_resolution", values="series"
    ).fillna(0)
    figure, axis = plt.subplots(figsize=(10, 6))
    image = axis.imshow(pivot.to_numpy(), cmap="YlGnBu", aspect="auto")
    axis.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            axis.text(
                column,
                row,
                f"{int(pivot.iloc[row, column])}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )
    axis.set_title("Native diagnostic series by spatial and temporal grain")
    figure.colorbar(image, ax=axis, label="Series")
    path = root / "overview" / "native_grain_matrix.png"
    _save(figure, path)
    records.append(
        {
            "path": path,
            "method_id": "series_profile",
            "series_id": None,
            "panel_id": None,
            "artifact_role": "native_grain_matrix",
        }
    )
    status = statistics.groupby(["method_id", "status"]).size().unstack(fill_value=0)
    figure, axis = plt.subplots(figsize=(12, max(6, 0.32 * len(status))))
    status.plot(
        kind="barh",
        stacked=True,
        ax=axis,
        color=["#2CA02C", "#F2C14E", "#D62728", "#7F7F7F"][: len(status.columns)],
    )
    axis.set_xlabel("Diagnostic statistics")
    axis.set_ylabel("")
    axis.set_title("Diagnostic execution and applicability")
    axis.legend(title="Status", loc="lower right")
    path = root / "overview" / "method_execution.png"
    _save(figure, path)
    records.append(
        {
            "path": path,
            "method_id": "series_profile",
            "series_id": None,
            "panel_id": None,
            "artifact_role": "method_execution",
        }
    )
    if not events.empty:
        event_counts = (
            events.groupby(["method_id", "event_type"])
            .size()
            .sort_values(ascending=False)
            .head(25)
            .sort_values()
        )
        figure, axis = plt.subplots(figsize=(12, 8))
        event_counts.plot(kind="barh", color="#8C6BB1", ax=axis)
        axis.set_xlabel("Detected events")
        axis.set_ylabel("")
        axis.set_title("Highest-volume diagnostic event types")
        path = root / "overview" / "diagnostic_events.png"
        _save(figure, path)
        records.append(
            {
                "path": path,
                "method_id": "robust_residual_anomaly",
                "series_id": None,
                "panel_id": None,
                "artifact_role": "event_summary",
            }
        )
    return records


def artifact_metadata(
    output_root: Path,
    diagnostic_run_id: str,
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        path = Path(record["path"]).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        width = None
        height = None
        media_type = "application/octet-stream"
        if path.suffix == ".png":
            media_type = "image/png"
            with Image.open(path) as image:
                width, height = image.size
        elif path.suffix == ".parquet":
            media_type = "application/vnd.apache.parquet"
        elif path.name.endswith(".csv.gz"):
            media_type = "text/csv+gzip"
        rows.append(
            {
                "artifact_id": str(
                    uuid.uuid5(
                        uuid.UUID(diagnostic_run_id), str(path.relative_to(output_root.resolve()))
                    )
                ),
                "diagnostic_run_id": diagnostic_run_id,
                "series_id": record.get("series_id"),
                "panel_id": record.get("panel_id"),
                "method_id": record.get("method_id"),
                "artifact_role": record["artifact_role"],
                "media_type": media_type,
                "absolute_path": str(path),
                "relative_path": str(path.relative_to(output_root.resolve())),
                "sha256": digest,
                "byte_count": path.stat().st_size,
                "width_pixels": width,
                "height_pixels": height,
                "created_at": datetime.now(UTC),
                "provenance": {
                    "diagnostic_run_id": diagnostic_run_id,
                    "method_id": record.get("method_id"),
                },
            }
        )
    return pd.DataFrame.from_records(rows)

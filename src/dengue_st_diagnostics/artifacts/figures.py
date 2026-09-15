from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pywt

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from dengue_st_diagnostics.diagnostics.spatial import _boundary_frame, _name

BLUE = "#276FBF"
BLUE_DARK = "#17324D"
BLUE_LIGHT = "#B9D7F2"
GOLD = "#D39A22"
GREY = "#D9E0E7"
INK = "#1F2933"


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GREY, linewidth=0.7)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _coverage(inventory: pd.DataFrame, path: Path) -> Path | None:
    if inventory.empty:
        return None
    matrix = inventory.pivot_table(
        index="spatial_resolution",
        columns="temporal_resolution",
        values="rows",
        aggfunc="sum",
        fill_value=0,
    )
    matrix = np.log10(matrix + 1)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    image = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xlabel("Temporal resolution")
    ax.set_ylabel("Spatial resolution")
    ax.set_title("Observed and validly aggregated records", fontsize=10, color=INK)
    fig.suptitle("Dengue data coverage by grain", fontsize=15, color=INK)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("log10(records + 1)")
    return _save(fig, path)


def _representative(multiscale: pd.DataFrame, spatial: str = "admin0") -> pd.DataFrame:
    selected = multiscale.loc[multiscale["spatial_resolution"].eq(spatial)]
    if selected.empty:
        return pd.DataFrame()
    sizes = selected.groupby("panel_id")["period_start"].nunique().sort_values(ascending=False)
    return selected.loc[selected["panel_id"].eq(sizes.index[0])].sort_values("period_start")


def _trend(multiscale: pd.DataFrame, path: Path) -> Path | None:
    panel = _representative(multiscale)
    if panel.empty:
        return None
    series = panel.groupby("period_start", as_index=False)["cases"].sum()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(series["period_start"], series["cases"], color=BLUE, linewidth=1.8)
    ax.fill_between(series["period_start"], series["cases"], color=BLUE_LIGHT, alpha=0.35)
    ax.set_ylabel("Reported cases")
    ax.set_xlabel("Period")
    resolution = panel["temporal_resolution"].iloc[0].title()
    source = panel["source"].iloc[0]
    ax.set_title(
        f"{resolution} observations; source {source}",
        fontsize=10,
        color=INK,
    )
    fig.suptitle("Indonesia dengue time series", fontsize=15, color=INK)
    _style(ax)
    return _save(fig, path)


def _wavelet(multiscale: pd.DataFrame, path: Path) -> Path | None:
    panel = _representative(multiscale)
    if panel.empty:
        return None
    series = panel.groupby("period_start")["cases"].sum().sort_index()
    if len(series) < 12:
        return None
    scales = np.arange(2, min(128, len(series) // 2) + 1)
    coefficients, _ = pywt.cwt(series.to_numpy() - series.mean(), scales, "cmor1.5-1.0")
    power = np.abs(coefficients) ** 2
    fig, ax = plt.subplots(figsize=(11, 5.5))
    image = ax.imshow(
        power,
        aspect="auto",
        origin="lower",
        extent=[0, len(series) - 1, scales[0], scales[-1]],
        cmap="Blues",
    )
    ticks = np.linspace(0, len(series) - 1, min(8, len(series))).astype(int)
    ax.set_xticks(ticks, [series.index[index].strftime("%Y-%m") for index in ticks], rotation=30)
    ax.set_xlabel("Period")
    ax.set_ylabel("Wavelet scale")
    ax.set_title(
        "Morlet power; darker bands indicate stronger periodic components", fontsize=10, color=INK
    )
    fig.suptitle("Time-varying dengue periodicity", fontsize=15, color=INK)
    fig.colorbar(image, ax=ax, label="Power")
    return _save(fig, path)


def _spatial_autocorrelation(global_spatial: pd.DataFrame, path: Path) -> Path | None:
    if global_spatial.empty:
        return None
    moran = global_spatial.loc[global_spatial["method"].eq("global_moran")]
    if moran.empty:
        return None
    panel_id = moran.groupby("panel_id").size().sort_values(ascending=False).index[0]
    selected = moran.loc[moran["panel_id"].eq(panel_id)].sort_values("period_start")
    fig, ax = plt.subplots(figsize=(11, 5.2))
    significant = selected["p_value"].lt(0.05)
    ax.plot(selected["period_start"], selected["estimate"], color=BLUE, linewidth=1.5)
    ax.scatter(
        selected.loc[significant, "period_start"],
        selected.loc[significant, "estimate"],
        color=GOLD,
        s=24,
        label="Permutation p < 0.05",
    )
    ax.axhline(0, color=BLUE_DARK, linewidth=0.8)
    ax.set_xlabel("Period")
    ax.set_ylabel("Moran's I")
    ax.set_title(f"Panel {panel_id}", fontsize=9, color=INK)
    ax.legend(frameon=False)
    fig.suptitle("Spatial autocorrelation through time", fontsize=15, color=INK)
    _style(ax)
    return _save(fig, path)


def _similarity(similarity: pd.DataFrame, path: Path) -> Path | None:
    if similarity.empty:
        return None
    counts = similarity.groupby("panel_id").size().sort_values(ascending=False)
    candidates = counts.loc[counts.le(1_225)]
    panel_id = candidates.index[0] if not candidates.empty else counts.index[-1]
    selected = similarity.loc[similarity["panel_id"].eq(panel_id)]
    names = sorted(set(selected["source_location"]) | set(selected["target_location"]))
    matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for row in selected.itertuples(index=False):
        matrix.loc[row.source_location, row.target_location] = row.spearman_similarity
        matrix.loc[row.target_location, row.source_location] = row.spearman_similarity
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    step = max(1, len(names) // 15)
    ticks = np.arange(0, len(names), step)
    ax.set_xticks(ticks, [names[index] for index in ticks], rotation=90, fontsize=7)
    ax.set_yticks(ticks, [names[index] for index in ticks], fontsize=7)
    ax.set_title("Pairwise Spearman correlation of area time series", fontsize=10, color=INK)
    fig.suptitle("Transfer-learning similarity", fontsize=15, color=INK)
    fig.colorbar(image, ax=ax, label="Correlation")
    return _save(fig, path)


def _readiness(readiness: pd.DataFrame, path: Path) -> Path | None:
    if readiness.empty:
        return None
    ranked = readiness.groupby("panel_id")["readiness_score"].mean().nlargest(20).index
    matrix = readiness.loc[readiness["panel_id"].isin(ranked)].pivot(
        index="panel_id",
        columns="use_case",
        values="readiness_score",
    )
    fig, ax = plt.subplots(figsize=(11, max(5, len(matrix) * 0.35)))
    image = ax.imshow(matrix, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=25, ha="right")
    labels = [value[:55] for value in matrix.index]
    ax.set_yticks(range(len(matrix.index)), labels, fontsize=7)
    ax.set_xlabel("Downstream use")
    ax.set_ylabel("Data panel")
    ax.set_title(
        "Scores combine period count, area count, zeros, and population coverage",
        fontsize=10,
        color=INK,
    )
    fig.suptitle("Modeling readiness by data grain", fontsize=15, color=INK)
    fig.colorbar(image, ax=ax, label="Readiness score")
    return _save(fig, path)


def _hotspot_map(
    emerging: pd.DataFrame,
    boundaries: dict[str, gpd.GeoDataFrame],
    path: Path,
) -> Path | None:
    if emerging.empty:
        return None
    panel_id = emerging.groupby("panel_id").size().sort_values(ascending=False).index[0]
    selected = emerging.loc[emerging["panel_id"].eq(panel_id)].copy()
    level = "adm2" if selected["location_name"].astype(str).str.count(r"\|").max() >= 1 else "adm1"
    geometry = _boundary_frame(level, boundaries)
    if geometry is None:
        return None
    selected["match_key"] = selected["location_name"].map(_name)
    joined = geometry.merge(selected[["match_key", "category"]], on="match_key", how="left")
    categories = sorted(value for value in joined["category"].dropna().unique())
    palette = [BLUE_DARK, BLUE, GOLD, "#D87542", "#7A8B55", "#A56A9D", BLUE_LIGHT]
    colors = {category: palette[index % len(palette)] for index, category in enumerate(categories)}
    joined["color"] = joined["category"].map(colors).fillna("#E8EDF2")
    fig, ax = plt.subplots(figsize=(12, 6))
    joined.plot(ax=ax, color=joined["color"], edgecolor="white", linewidth=0.25)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=colors[value],
            label=value,
            markersize=8,
        )
        for value in categories
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8)
    ax.set_axis_off()
    ax.set_title(f"Getis-Ord trajectories; panel {panel_id}", fontsize=9, color=INK)
    fig.suptitle("Emerging dengue hotspot classification", fontsize=15, color=INK)
    return _save(fig, path)


def create_figures(
    output: Path,
    inventory: pd.DataFrame,
    multiscale: pd.DataFrame,
    global_spatial: pd.DataFrame,
    spatiotemporal: dict[str, pd.DataFrame],
    boundaries: dict[str, gpd.GeoDataFrame],
) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    candidates = [
        _coverage(inventory, output / "coverage_by_grain.png"),
        _trend(multiscale, output / "national_time_series.png"),
        _wavelet(multiscale, output / "wavelet_power.png"),
        _spatial_autocorrelation(global_spatial, output / "spatial_autocorrelation.png"),
        _similarity(spatiotemporal["transfer_similarity"], output / "transfer_similarity.png"),
        _readiness(spatiotemporal["modeling_readiness"], output / "modeling_readiness.png"),
        _hotspot_map(
            spatiotemporal["emerging_hotspots"], boundaries, output / "emerging_hotspots.png"
        ),
    ]
    return [path for path in candidates if path is not None]


def _puskesmas_monthly(frame: pd.DataFrame, path: Path) -> Path | None:
    selected = frame.loc[
        frame["aggregation_level"].eq("puskesmas") & frame["temporal_resolution"].eq("month")
    ].copy()
    if selected.empty:
        return None
    matrix = selected.pivot_table(
        index="puskesmas",
        columns="period_start",
        values="cases",
        aggfunc="sum",
    )
    complete_months = pd.date_range(matrix.columns.min(), matrix.columns.max(), freq="MS")
    matrix = matrix.reindex(columns=complete_months)
    fig, ax = plt.subplots(figsize=(13, max(4.5, len(matrix) * 0.65)))
    image = ax.imshow(matrix, cmap="Blues", aspect="auto", interpolation="nearest")
    step = max(1, len(matrix.columns) // 12)
    ticks = np.arange(0, len(matrix.columns), step)
    ax.set_xticks(ticks, [matrix.columns[index].strftime("%Y-%m") for index in ticks], rotation=45)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xlabel("Month")
    ax.set_ylabel("Puskesmas")
    ax.set_title("Blank cells mean no public report was found", fontsize=10, color=INK)
    fig.suptitle("Public monthly dengue reports by Puskesmas", fontsize=15, color=INK)
    fig.colorbar(image, ax=ax, label="Reported dengue cases")
    return _save(fig, path)


def _puskesmas_annual(frame: pd.DataFrame, path: Path) -> Path | None:
    selected = frame.loc[
        frame["aggregation_level"].eq("puskesmas") & frame["temporal_resolution"].eq("year")
    ].copy()
    if selected.empty:
        return None
    selected["year"] = selected["period_start"].dt.year
    matrix = selected.pivot_table(
        index="puskesmas",
        columns="year",
        values="cases",
        aggfunc="sum",
    )
    fig, ax = plt.subplots(figsize=(11, max(5, len(matrix) * 0.32)))
    image = ax.imshow(matrix, cmap="Blues", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=7)
    ax.set_xlabel("Year")
    ax.set_ylabel("Puskesmas")
    ax.set_title("Annual reports from Purbalingga", fontsize=10, color=INK)
    fig.suptitle("Long-term public dengue history by Puskesmas", fontsize=15, color=INK)
    fig.colorbar(image, ax=ax, label="Reported dengue cases")
    return _save(fig, path)


def create_puskesmas_figures(output: Path, frame: pd.DataFrame) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        return []
    candidates = [
        _puskesmas_monthly(frame, output / "puskesmas_monthly_coverage.png"),
        _puskesmas_annual(frame, output / "puskesmas_annual_history.png"),
    ]
    return [path for path in candidates if path is not None]


def _vector_map(frame: pd.DataFrame, path: Path) -> Path | None:
    if frame.empty:
        return None
    selected = frame.loc[frame["latitude"].notna() & frame["longitude"].notna()].copy()
    if selected.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True, sharey=True)
    groups = {
        "Dengue vectors": selected.loc[selected["requested_species"].str.startswith("Aedes")],
        "Malaria vectors": selected.loc[selected["requested_species"].str.startswith("Anopheles")],
    }
    for ax, (title, group) in zip(axes, groups.items(), strict=True):
        for species, records in group.groupby("requested_species"):
            ax.scatter(
                records["longitude"],
                records["latitude"],
                s=7,
                alpha=0.45,
                label=species.replace("Aedes ", "Ae. ").replace("Anopheles ", "An. "),
            )
        ax.set_title(title, color=INK)
        ax.set_xlabel("Longitude")
        ax.set_xlim(94, 142)
        ax.set_ylim(-12, 8)
        ax.legend(frameon=False, fontsize=6, ncol=2)
        _style(ax)
    axes[0].set_ylabel("Latitude")
    fig.suptitle("Open GBIF vector occurrence records in Indonesia", fontsize=15, color=INK)
    return _save(fig, path)


def _sequence_sampling(frame: pd.DataFrame, path: Path) -> Path | None:
    if frame.empty:
        return None
    selected = frame.copy()
    selected["year"] = pd.to_numeric(
        selected["collection_date"].astype("string").str.extract(r"(19\d{2}|20\d{2})")[0],
        errors="coerce",
    )
    selected = selected.loc[selected["year"].between(1970, 2030)]
    if selected.empty:
        return None
    counts = selected.groupby(["year", "classification"]).size().unstack(fill_value=0)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, disease in zip(axes, ["dengue", "malaria"], strict=True):
        labels = selected.loc[selected["disease"].eq(disease), "classification"].unique()
        counts.reindex(columns=labels, fill_value=0).plot(ax=ax, linewidth=1.4)
        ax.set_ylabel("Submitted sequences")
        ax.set_title(disease.title(), color=INK)
        ax.legend(frameon=False, fontsize=7, ncol=5)
        _style(ax)
    axes[-1].set_xlabel("Collection year")
    fig.suptitle("Indonesia pathogen sequence sampling over time", fontsize=15, color=INK)
    return _save(fig, path)


def _malaria_indicators(frame: pd.DataFrame, path: Path) -> Path | None:
    if frame.empty:
        return None
    codes = ["MALARIA_CONF_CASES", "MALARIA_PF_INDIG", "MALARIA_PV_INDIG"]
    selected = frame.loc[frame["indicator_code"].isin(codes)].copy()
    if selected.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, group in selected.groupby("indicator_name"):
        group = group.sort_values("year")
        ax.plot(group["year"], group["value"], marker="o", linewidth=1.7, label=name)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cases")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.suptitle("WHO-reported Indonesia malaria indicators", fontsize=15, color=INK)
    return _save(fig, path)


def create_surveillance_figures(
    output: Path,
    vectors: pd.DataFrame,
    sequences: pd.DataFrame,
    who: pd.DataFrame,
) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    candidates = [
        _vector_map(vectors, output / "vector_occurrences.png"),
        _sequence_sampling(sequences, output / "pathogen_sequence_sampling.png"),
        _malaria_indicators(who, output / "who_malaria_indicators.png"),
    ]
    return [path for path in candidates if path is not None]

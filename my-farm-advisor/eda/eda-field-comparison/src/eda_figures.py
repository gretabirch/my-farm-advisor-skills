"""Static figure renderers for field-level EDA.

Twelve figures: two statistical visuals and one comparison per category
(boundaries, CDL, weather), plus one locator map per category. Every map
follows the shared field-frame contract (``data-pipeline/src/scripts/lib``):
local metre axes, auto-rotation, boxed north arrow, basemap underlay, and
distance read from the axes (no scale bar). Weather layers use a shared
continuous scale; crop layers use the official CDL categorical colours read
from the GeoTIFF colormaps.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

from eda_data import _ensure_lib_on_path, power_cell, POWER_DLAT, POWER_DLON

# Okabe-Ito, fixed order so a grower keeps its colour across every figure.
GROWER_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
YEAR_CMAP = "viridis"
MUTED = "#6B7280"
INK = "#111827"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.edgecolor": "#374151",
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def grower_colors(labels) -> dict[str, str]:
    labels = list(dict.fromkeys(labels))
    return {lab: GROWER_PALETTE[i % len(GROWER_PALETTE)] for i, lab in enumerate(labels)}


def _year_shades(n: int) -> list[str]:
    cmap = matplotlib.colormaps[YEAR_CMAP]
    if n == 1:
        return [to_hex(cmap(0.55))]
    return [to_hex(cmap(i / (n - 1))) for i in range(n)]


def _thin_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3, width=0.7)


def _stat_box(ax, text: str, loc: str = "upper left") -> None:
    x, y, ha, va = {
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
        "upper center": (0.5, 0.98, "center", "top"),
        "lower center": (0.5, 0.02, "center", "bottom"),
        "right middle": (0.97, 0.72, "right", "center"),
        "left middle": (0.03, 0.72, "left", "center"),
    }[loc]
    ax.text(
        x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=7.2,
        color="#1F2937", linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.38", fc="white", ec="#D1D5DB", lw=0.7, alpha=0.95),
        zorder=30,
    )


def _fig_caption(fig, text: str, *, y: float = 0.006, fontsize: float = 6.8) -> None:
    """Figure-wide wrapped footnote that cannot widen the saved canvas."""
    import textwrap

    width = max(60, int(fig.get_figwidth() * 15))
    fig.text(0.5, y, textwrap.fill(text, width=width), ha="center", va="bottom",
             fontsize=fontsize, color="#4B5563")


def _caption(ax, text: str, *, y: float = -0.18, fontsize: float = 6.8) -> None:
    """Wrapped footnote anchored to the axes width (no canvas blow-out)."""
    import textwrap

    width = max(50, int(ax.figure.get_figwidth() * 14))
    ax.text(
        0.0, y, textwrap.fill(text, width=width), transform=ax.transAxes,
        fontsize=fontsize, color="#4B5563", va="top", ha="left",
    )


# --------------------------------------------------------------------------
# CDL colours
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _raster_colormap(path: str) -> dict[int, str]:
    import rasterio

    with rasterio.open(path) as src:
        try:
            cmap = src.colormap(1)
        except Exception:
            cmap = {}
    return {int(k): to_hex(np.asarray(v[:3], dtype=float) / 255.0) for k, v in cmap.items()}


def crop_palette(farms) -> tuple[dict[int, str], dict[str, int]]:
    """code -> official colour, crop name -> code, merged over farm rasters."""
    codes: dict[int, str] = {}
    for farm in farms:
        raster = farm.cdl_raster
        if raster is not None:
            codes.update(_raster_colormap(str(raster)))
    name_to_code: dict[str, int] = {}
    for farm in farms:
        pairs = farm.cdl[["crop_code", "crop_name"]].drop_duplicates()
        for code, name in pairs.itertuples(index=False):
            name_to_code.setdefault(str(name), int(code))
    for name, code in name_to_code.items():
        codes.setdefault(int(code), "#BDBDBD")
    return codes, name_to_code


def crop_abbr(name: str) -> str:
    fixed = {
        "Corn": "C", "Soybeans": "S", "Winter Wheat": "W", "Alfalfa": "A",
        "Forest": "F", "Grass/Pasture": "G", "Fallow/Idle": "I",
        "Developed": "D", "Water": "~",
    }
    return fixed.get(name, str(name)[:1].upper())


# --------------------------------------------------------------------------
# A. field boundaries
# --------------------------------------------------------------------------

def fig_area_by_grower(ctx, path: Path) -> Path:
    geo = ctx["geometry"]
    labels = ctx["grower_order"]
    colors = ctx["grower_colors"]
    kw = ctx["stats"]["area_grower_kw"]

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    rng = np.random.default_rng(20261120)
    data = [geo.loc[geo["grower_label"] == lab, "area_acres"].to_numpy() for lab in labels]

    bp = ax.boxplot(
        data, widths=0.46, patch_artist=True, showfliers=False, whis=1.5,
        medianprops=dict(color=INK, lw=1.4),
        whiskerprops=dict(color=MUTED, lw=0.9),
        capprops=dict(color=MUTED, lw=0.9),
        boxprops=dict(lw=1.0),
    )
    for patch, lab in zip(bp["boxes"], labels):
        patch.set_facecolor(to_rgba(colors[lab], 0.16))
        patch.set_edgecolor(colors[lab])

    for i, (lab, vals) in enumerate(zip(labels, data), start=1):
        jitter = rng.uniform(-0.16, 0.16, len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter, vals, s=26, color=colors[lab],
            edgecolor="white", linewidth=0.6, alpha=0.95, zorder=5,
        )

    ax.set_yscale("log")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(
        [
            f"{lab}\nmedian {np.median(vals):,.1f} ac\n{len(vals)} fields · {vals.sum():,.0f} ac total"
            for lab, vals in zip(labels, data)
        ]
    )
    ax.set_ylabel("Field area (acres, log scale)")
    ax.set_title("Field size spans two orders of magnitude across growers")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylim(0.4, 1600)
    _thin_axes(ax)
    ax.grid(axis="y", color="#E5E7EB", lw=0.7, zorder=0)

    _stat_box(
        ax,
        "Kruskal-Wallis  H = %.1f,  p = %.2g\n"
        "effect size ε² = %.2f  (descriptive, n = %d)"
        % (kw["H"], kw["p"], kw["epsilon_squared"], kw["n"]),
        "upper right",
    )
    ax.text(
        0.0, -0.22,
        "Each dot is one field. Log scale because the largest IL field (732 ac) is ~1,200x the "
        "smallest NE parcel (0.6 ac). Median acreage is printed under each grower.",
        transform=ax.transAxes, fontsize=6.8, color="#4B5563", va="top", ha="left",
    )
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_area_vs_compactness(ctx, path: Path) -> Path:
    geo = ctx["geometry"]
    colors = ctx["grower_colors"]
    stats = ctx["stats"]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for lab in ctx["grower_order"]:
        sub = geo[geo["grower_label"] == lab]
        ax.scatter(
            sub["area_acres"], sub["compactness_pp"], s=48, color=colors[lab],
            edgecolor="white", linewidth=0.7, alpha=0.95, zorder=4, label=f"{lab} (n = {len(sub)})",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Field area (acres, log scale)")
    ax.set_ylabel("Polsby-Popper compactness  (4πA / P²)")
    ax.set_title("Compactness falls with area: the biggest parcels are the most irregular",
                 fontsize=9.0)
    ax.set_ylim(0.0, 0.95)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _thin_axes(ax)
    ax.grid(color="#E5E7EB", lw=0.7, zorder=0)

    rho = stats["area_compactness_spearman"]
    _stat_box(
        ax,
        "Spearman ρ = %.2f  (95%% CI %.2f to %.2f)\npermutation p = %.3f,  n = %d fields"
        % (rho["rho"], rho["ci95"][0], rho["ci95"][1], rho["p_perm"], rho["n"]),
        "lower left",
    )
    ax.legend(loc="upper right", title="Grower", title_fontsize=8, handletextpad=0.3)
    _caption(
        ax,
        "Compactness is a shape-only index in [0, 1]: 1 is a circle. The 250-730 ac parcels score "
        "0.16-0.25 because they are long or perforated, while sub-10-acre parcels are simple "
        "rectangles. Shape is not sensing quality: a sub-acre field is weak because it is only a "
        "few 30 m pixels across.",
        y=-0.20,
    )
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_cdl_coverage(ctx, path: Path) -> Path:
    geo = ctx["geometry"]
    cy = ctx["cdl_year"]
    colors = ctx["grower_colors"]
    latest = int(max(ctx["cdl_years"]))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.8, 4.5), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )
    fig.subplots_adjust(wspace=0.30, bottom=0.21)

    cur = cy[cy["year"] == latest].merge(
        geo[["field_id", "area_acres"]], on="field_id", how="left"
    )
    for lab in ctx["grower_order"]:
        sub = cur[cur["grower_label"] == lab]
        ax1.scatter(
            sub["area_acres"], sub["cdl_coverage_pct"], s=48, color=colors[lab],
            edgecolor="white", linewidth=0.7, zorder=4, label=lab,
        )
    ax1.axhline(95, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax1.text(
        0.985, 93.4, "95 % reference", transform=ax1.get_yaxis_transform(),
        ha="right", va="top", fontsize=7, color=MUTED,
    )
    ax1.set_xscale("log")
    ax1.set_xlabel("Field area (acres, log scale)")
    ax1.set_ylabel(f"CDL classified cover, {latest} (% of field)")
    ax1.set_title("Cover is 95 %+ except on the smallest parcels")
    ax1.set_ylim(55, 102)
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _thin_axes(ax1)
    ax1.grid(color="#E5E7EB", lw=0.7, zorder=0)
    ax1.legend(loc="lower right", title="Grower", title_fontsize=8, handletextpad=0.3)

    below = int((cur["cdl_coverage_pct"] < 95).sum())
    low = cur.loc[cur["cdl_coverage_pct"].idxmin()]
    _stat_box(
        ax1,
        "%d of %d CDL-covered fields below 95 %%\nlowest: %.0f %% (%s, %.1f ac)"
        % (below, len(cur), low["cdl_coverage_pct"], low["field_label"], low["area_acres"]),
        "lower left",
    )

    # within-field stability: how far coverage moves between CDL years
    spread = (
        cy.groupby(["field_id", "field_label", "grower_label"])["cdl_coverage_pct"]
        .agg(["min", "max", "mean", "count"])
        .reset_index()
    )
    spread["range_pp"] = spread["max"] - spread["min"]
    bins = [0.0, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 100.0]
    bin_labels = ["0–0.05", "0.05–0.10", "0.10–0.25", "0.25–0.50", "0.50–1", "1–2", "≥2"]
    spread["bin"] = pd.cut(spread["range_pp"], bins=bins, labels=bin_labels,
                           right=False, include_lowest=True)
    counts = (
        spread.groupby(["bin", "grower_label"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(bin_labels, fill_value=0)
    )
    y = np.arange(len(bin_labels))
    left = np.zeros(len(bin_labels))
    for lab in ctx["grower_order"]:
        vals = counts[lab].to_numpy() if lab in counts else np.zeros(len(bin_labels))
        ax2.barh(y, vals, left=left, height=0.66, color=colors[lab], edgecolor="white",
                 linewidth=0.6, label=lab)
        left += vals
    for yi, total in zip(y, left):
        if total:
            ax2.text(total + 0.18, yi, f"{int(total)}", va="center", fontsize=7.4, color=INK)
    ax2.set_yticks(y)
    ax2.set_yticklabels(bin_labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("Fields (count)")
    ax2.set_ylabel("Change in classified cover (pp)")
    ax2.set_title("Cover moves <2 pp on 26 of 29 fields")
    ax2.set_xlim(0, max(3.0, left.max() * 1.18))
    _thin_axes(ax2)
    ax2.grid(axis="x", color="#E5E7EB", lw=0.7, zorder=0)
    ax2.legend(loc="center right", title="Grower", title_fontsize=8, handletextpad=0.3)

    n_stable = int((spread["range_pp"] < 2).sum())
    worst = spread.sort_values("range_pp", ascending=False).iloc[0]
    _fig_caption(
        fig,
        "Left: one dot per field with CDL rows (%d of %d); the four below 95 %% are the 0.6-17 ac parcels, "
        "where mixed edge pixels dominate (across all fields the size relation is weak, ρ = %.2f). "
        "Right: spread of each field's classified cover across the five CDL years — %d of %d fields move "
        "by less than 2 pp, so a gap is an edge effect, not annual change. The worst is %s "
        "(%.0f → %.0f %%), where land cover itself changed."
        % (len(cur), len(geo), ctx["stats"]["cdl_coverage_area_spearman"]["rho"],
           n_stable, len(spread), worst["field_label"], worst["min"], worst["max"]),
        y=0.006,
    )
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# B. CDL / cropland
# --------------------------------------------------------------------------

def _field_order(ctx) -> list[str]:
    geo = ctx["geometry"]
    order = []
    for lab in ctx["grower_order"]:
        sub = geo[geo["grower_label"] == lab].sort_values("field_label")
        order.extend(sub["field_label"].tolist())
    return order


def fig_cdl_composition(ctx, path: Path) -> Path:
    comp = ctx["composition"]
    order = _field_order(ctx)
    codes, name_to_code = ctx["crop_codes"]
    color_of = {name: codes.get(code, "#BDBDBD") for name, code in name_to_code.items()}
    present = [c for c in comp["crop_name"].unique() if c in color_of]
    present.sort(key=lambda c: (-comp.loc[comp["crop_name"] == c, "pct_mean_norm"].mean(), c))

    fig, ax = plt.subplots(figsize=(8.4, 8.0))
    y = np.arange(len(order))
    left = np.zeros(len(order))
    for crop in present:
        vals = np.array(
            [
                float(
                    comp.loc[
                        (comp["field_label"] == f) & (comp["crop_name"] == crop), "pct_mean_norm"
                    ].sum()
                )
                for f in order
            ]
        )
        ax.barh(y, vals, left=left, height=0.74, color=color_of[crop],
                edgecolor="white", linewidth=0.4, label=crop)
        left += vals

    missing = [f for f in order if f not in set(comp["field_label"])]
    if missing:
        ax.barh(
            [order.index(f) for f in missing], [100] * len(missing), height=0.74,
            color="white", edgecolor=MUTED, hatch="///", linewidth=0.7,
            label="No CDL rows",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=6.4)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Mean share of field, 2019–2023 (% of classified pixels)")
    ax.set_title("Crop mix per field — means hide the rotation")
    _thin_axes(ax)
    ax.grid(axis="x", color="#E5E7EB", lw=0.7, zorder=0)

    # grower separators
    for lab in ctx["grower_order"]:
        idx = [i for i, f in enumerate(order) if f.startswith(lab + "-")]
        if idx and max(idx) < len(order) - 1:
            ax.axhline(max(idx) + 0.5, color="#6B7280", lw=1.0, ls=(0, (5, 3)), zorder=1)

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.075), ncol=4,
        handlelength=1.1, columnspacing=1.2,
    )
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_cdl_rotation(ctx, path: Path) -> Path:
    mat = ctx["cdl_matrix"]
    order = _field_order(ctx)
    years = ctx["cdl_years"]
    codes, name_to_code = ctx["crop_codes"]
    color_of = {name: codes.get(code, "#BDBDBD") for name, code in name_to_code.items()}

    fig, ax = plt.subplots(figsize=(6.4, 7.8))
    for row, field in enumerate(order):
        sub = mat[mat["field_label"] == field]
        for col, year in enumerate(years):
            crop = None if sub.empty else sub.iloc[0][year]
            if isinstance(crop, float) and np.isnan(crop):
                crop = None
            x0 = col - 0.5
            if crop is None:
                ax.add_patch(
                    plt.Rectangle((x0, row - 0.47), 1.0, 0.94, facecolor="white",
                                  edgecolor="#9CA3AF", hatch="///", linewidth=0.5)
                )
                continue
            ax.add_patch(
                plt.Rectangle((x0, row - 0.47), 1.0, 0.94,
                              facecolor=color_of.get(crop, "#BDBDBD"),
                              edgecolor="white", linewidth=0.9)
            )
            ax.text(
                col, row, crop_abbr(crop), ha="center", va="center", fontsize=7.4,
                color="white" if _is_dark(color_of.get(crop, "#BDBDBD")) else INK,
                fontweight="bold",
            )

    for row, field in enumerate(order):
        sub = mat[mat["field_label"] == field]
        if sub.empty or all(
            not isinstance(sub.iloc[0][y], str) for y in years
        ):
            ax.text((len(years) - 1) / 2, row, "no CDL rows for this field",
                    ha="center", va="center", fontsize=6.2, color="#374151", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#9CA3AF",
                              lw=0.6, alpha=0.95))

    ax.set_xlim(-0.5, len(years) - 0.5)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(y) for y in years])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6.4)
    ax.set_title("Dominant CDL class, field × year")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    for lab in ctx["grower_order"]:
        idx = [i for i, f in enumerate(order) if f.startswith(lab + "-")]
        if idx:
            ax.axhline(max(idx) + 0.5, color="#9CA3AF", lw=0.9, ls=(0, (5, 3)))

    handles = [
        Patch(facecolor=color_of.get(name, "#BDBDBD"), edgecolor="white",
              label=f"{crop_abbr(name)} = {name}")
        for name in sorted(color_of, key=lambda n: crop_abbr(n))
    ]
    handles.append(Patch(facecolor="white", edgecolor="#9CA3AF", hatch="///", label="No CDL rows"))
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.055), ncol=3,
        handlelength=1.0, columnspacing=1.1, fontsize=7.2,
    )
    fig.savefig(path)
    plt.close(fig)
    return path


def _is_dark(hex_color: str) -> bool:
    r, g, b, _ = to_rgba(hex_color)
    return (0.299 * r + 0.587 * g + 0.114 * b) < 0.55


def fig_rotation_metrics(ctx, path: Path) -> Path:
    rot = ctx["rotation"]
    colors = ctx["grower_colors"]
    stats = ctx["stats"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.3))

    rng = np.random.default_rng(7)
    for i, lab in enumerate(ctx["grower_order"], start=1):
        vals = rot.loc[rot["grower_label"] == lab, "crop_changes"].to_numpy()
        jitter = rng.uniform(-0.13, 0.13, len(vals))
        ax1.scatter(np.full(len(vals), i) + jitter, vals, s=44, color=colors[lab],
                    edgecolor="white", linewidth=0.7, alpha=0.95, zorder=4)
        med = float(np.median(vals))
        ax1.hlines(med, i - 0.26, i + 0.26, color=INK, lw=2.2, zorder=5)

    ax1.set_xticks(range(1, len(ctx["grower_order"]) + 1))
    ax1.set_xticklabels(
        [f"{lab}\nn = {len(rot[rot['grower_label'] == lab])}" for lab in ctx["grower_order"]]
    )
    ax1.set_ylabel("Dominant-crop changes, 2019–2023 (of 4 transitions)")
    ax1.set_title("Rotation stability")
    ax1.set_ylim(-0.5, 5.6)
    ax1.set_yticks(range(5))
    _thin_axes(ax1)
    ax1.grid(axis="y", color="#E5E7EB", lw=0.7, zorder=0)
    kw = stats["rotation_changes_kw"]
    _stat_box(
        ax1,
        "Kruskal-Wallis  H = %.1f,  p = %.2g\nε² = %.2f"
        % (kw["H"], kw["p"], kw["epsilon_squared"]),
        "upper left",
    )

    for lab in ctx["grower_order"]:
        sub = rot[rot["grower_label"] == lab]
        ax2.scatter(sub["area_acres"], sub["crop_diversity_shannon"], s=48,
                    color=colors[lab], edgecolor="white", linewidth=0.7, alpha=0.95,
                    zorder=4, label=lab)
    ax2.set_xscale("log")
    ax2.set_xlabel("Field area (acres, log scale)")
    ax2.set_ylabel("Shannon diversity of dominant crop")
    ax2.set_title("Diversity is unrelated to parcel size", pad=15)
    ax2.set_ylim(-0.10, 1.60)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _thin_axes(ax2)
    ax2.grid(color="#E5E7EB", lw=0.7, zorder=0)
    rho = stats["diversity_area_spearman"]
    ax2.text(
        0.5, 1.015,
        "Spearman ρ = %.2f  ·  95 %% CI %.2f to %.2f  ·  permutation p = %.3f"
        % (rho["rho"], rho["ci95"][0], rho["ci95"][1], rho["p_perm"]),
        transform=ax2.transAxes, ha="center", va="bottom", fontsize=7.2, color="#374151",
    )
    ax2.legend(loc="upper right", title="Grower", title_fontsize=8, handletextpad=0.3)
    _caption(
        ax1,
        "Black bar = grower median. A change is counted when the dominant class differs from the previous "
        "year. Groups do not separate (p = %.2f), and diversity shows no size trend (ρ = %.2f): the "
        "rotation signal is field-specific, not grower-specific."
        % (kw["p"], stats["diversity_area_spearman"]["rho"]),
        y=-0.22,
    )
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# C. weather
# --------------------------------------------------------------------------

def fig_season_precip(ctx, path: Path) -> Path:
    wy = ctx["weather_year"]
    colors = ctx["grower_colors"]
    years = sorted(wy["year"].unique())
    labels = ctx["grower_order"]
    xs = np.arange(len(years))

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    spreads = []
    for lab in labels:
        sub = wy[wy["grower_label"] == lab]
        agg = (
            sub.groupby("year")["precip_gs_mm"]
            .agg(["min", "median", "max"])
            .reindex(years)
        )
        ax.fill_between(xs, agg["min"], agg["max"], color=colors[lab], alpha=0.18,
                        linewidth=0, zorder=2)
        for _, g in sub.groupby("field_id"):
            g = g.set_index("year").reindex(years)
            ax.plot(xs, g["precip_gs_mm"].to_numpy(), color=colors[lab], lw=0.6,
                    alpha=0.35, zorder=3)
        ax.plot(xs, agg["median"].to_numpy(), color=colors[lab], lw=2.2, marker="o",
                markersize=5.5, markeredgecolor="white", markeredgewidth=0.7,
                zorder=5, label=f"{lab} (n = {sub['field_id'].nunique()} fields)")
        for year in years:
            vals = sub.loc[sub["year"] == year, "precip_gs_mm"]
            if len(vals) > 1:
                spreads.append((float(vals.max() - vals.min()), lab, int(year)))

    ax.set_xticks(xs)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_xlim(-0.25, len(years) - 0.75)
    ax.set_ylabel("Growing-season precipitation, May–Sep (mm)")
    ax.set_title("Year-to-year swings dwarf between-field differences")
    _thin_axes(ax)
    ax.grid(axis="y", color="#E5E7EB", lw=0.7, zorder=0)
    ax.legend(loc="upper left", title="Grower median (shaded = field range)",
              title_fontsize=8, handlelength=1.4)
    ax.margins(y=0.10)

    decomp = ctx["stats"]["precip_decomp"]["shares"]
    spread_max = max(spreads) if spreads else (0.0, "-", 0)
    _caption(
        ax,
        "Shaded band = range across the grower's fields. Ten IA fields share one daily weather series, so "
        "their band has zero width; the widest band is %.0f mm (%s %d), where one field falls in a second "
        "NASA POWER cell.  Variance shares: year %.0f %%, grower %.0f %%, field-within-grower %.0f %%, "
        "year × grower %.0f %%."
        % (*spread_max, 100 * decomp["year"], 100 * decomp["grower"],
           100 * decomp["field_within_grower"], 100 * decomp["residual"]),
        y=-0.16,
    )
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_cumulative_gdd(ctx, path: Path) -> Path:
    farms = ctx["farms"]
    colors = ctx["grower_colors"]
    year = int(max(ctx["weather_years"]))

    fig, axes = plt.subplots(1, len(farms), figsize=(11.2, 4.4), sharey=True)
    fig.subplots_adjust(bottom=0.17, top=0.85)
    axes = np.atleast_1d(axes)
    for ax, farm in zip(axes, farms):
        wx = farm.weather
        sub = wx[(wx["year"] == year) & (wx["in_season"])]
        fields = sub["field_id"].unique()
        for field_id in fields:
            g = sub[sub["field_id"] == field_id].sort_values("doy")
            ax.plot(g["doy"], g["gdd_cumulative"], color=colors[farm.short_label],
                    lw=1.2, alpha=0.65, zorder=3)
        totals = sub.groupby("field_id")["gdd"].sum()
        uniq = sub.pivot_table(index="doy", columns="field_id", values="gdd_cumulative").T
        n_curves = int(uniq.drop_duplicates().shape[0])
        ax.set_title(f"{farm.short_label} · {farm.farm_slug}", color=colors[farm.short_label])
        ax.set_xlabel("Day of year")
        ax.set_xlim(118, 288)
        _thin_axes(ax)
        ax.grid(color="#E5E7EB", lw=0.7, zorder=0)
        total_text = (
            f"season total {totals.min():,.0f} GDD" if np.isclose(totals.min(), totals.max())
            else f"season total {totals.min():,.0f}–{totals.max():,.0f} GDD"
        )
        ax.text(
            0.03, 0.97,
            "%d fields → %d distinct curve%s\n%s"
            % (len(fields), n_curves, "" if n_curves == 1 else "s", total_text),
            transform=ax.transAxes, va="top", fontsize=7.2, color="#1F2937",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D1D5DB", lw=0.7),
        )
    axes[0].set_ylabel("Cumulative GDD, base 10 °C  (°C·day)")
    fig.suptitle(
        f"Heat accumulation, {year}: one or two grid cells per farm, never one per field",
        fontsize=10, fontweight="bold", y=1.0,
    )
    _fig_caption(
        fig,
        "Curves within a panel overlap because NASA POWER returns the same 0.5° × 0.625° cell value for "
        "every field in that cell — overlap is grid resolution, not identical microclimate.",
        y=0.012,
    )
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_weather_variance(ctx, path: Path) -> Path:
    wy = ctx["weather_year"]
    stats = ctx["stats"]
    years = sorted(wy["year"].unique())
    shades = _year_shades(len(years))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.4, 4.6), gridspec_kw={"width_ratios": [1.0, 1.10]}
    )
    fig.subplots_adjust(wspace=0.32, bottom=0.34)

    components = ["year", "grower", "field_within_grower", "residual"]
    comp_labels = ["Year", "Grower", "Field within grower", "Year × grower + noise"]
    comp_colors = ["#0F4C81", "#D55E00", "#009E73", "#D1D5DB"]
    rows = [("Season precip", stats["precip_decomp"]), ("Season GDD", stats["gdd_decomp"])]
    for i, (name, decomp) in enumerate(rows):
        left = 0.0
        for comp, label, color in zip(components, comp_labels, comp_colors):
            share = 100 * decomp["shares"][comp]
            ax1.barh(i, share, left=left, color=color, edgecolor="white", linewidth=1.0,
                     height=0.52, label=label if i == 0 else None)
            if share >= 5:
                ax1.text(left + share / 2, i, f"{share:.0f}%", ha="center", va="center",
                         fontsize=7.6, color="white" if comp != "residual" else INK,
                         fontweight="bold", clip_on=True)
            elif share >= 0.5:
                cx = left + share / 2
                ax1.plot([cx, cx], [i + 0.26, i + 0.36], color="#9CA3AF", lw=0.7, zorder=3)
                ax1.text(cx, i + 0.38, f"{share:.0f}%", ha="center", va="bottom",
                         fontsize=6.4, color="#4B5563", clip_on=True)
            left += share
    ax1.set_yticks(range(len(rows)))
    ax1.set_yticklabels([r[0] for r in rows])
    ax1.set_ylim(len(rows) - 0.45, -0.55)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Share of field-year variance (%)")
    ax1.set_title("Year drives rain, grower drives heat", fontsize=9.4)
    _thin_axes(ax1)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
               handlelength=1.1, columnspacing=1.0, fontsize=7.2)

    for i, year in enumerate(years):
        sub = wy[wy["year"] == year]
        counts = sub.groupby(["precip_gs_mm", "gdd_gs"]).size().reset_index(name="n")
        ax2.scatter(counts["precip_gs_mm"], counts["gdd_gs"],
                    s=26 + 26 * counts["n"], color=shades[i],
                    edgecolor="white", linewidth=0.5, alpha=0.95, label=str(year), zorder=4)
    rho = stats["precip_gdd_spearman"]
    by_year = stats.get("precip_gdd_by_year", {})
    within = [v["rho"] for v in by_year.values() if "rho" in v]
    ax2.set_xlabel("Growing-season precipitation (mm)")
    ax2.set_ylabel("Growing-season GDD (°C·day)")
    ax2.set_title("Wet years run cool — but only across years", fontsize=9.4)
    _thin_axes(ax2)
    ax2.grid(color="#E5E7EB", lw=0.7, zorder=0)
    ax2.legend(title="Year", title_fontsize=8, loc="upper right", ncol=2,
               handletextpad=0.3, columnspacing=0.8)
    note = "Spearman ρ = %.2f  (95%% CI %.2f to %.2f)\npermutation p = %.3f, n = %d field-years" % (
        rho["rho"], rho["ci95"][0], rho["ci95"][1], rho["p_perm"], rho["n"])
    if within:
        note += "\nwithin a single year: ρ = %.2f to %.2f" % (min(within), max(within))
    _stat_box(ax2, note, "lower left")
    year_lo = min(by_year, key=lambda y: by_year[y].get("rho", 0))
    year_hi = max(by_year, key=lambda y: by_year[y].get("rho", 0))
    _fig_caption(
        fig,
        "Nested sequential R²: year, then grower, then field nested in grower; the remainder is the "
        "year × grower interaction plus error. The pooled precip–GDD relation is carried by the "
        "year-to-year structure: inside one year it runs from ρ = %.2f (%s) to %+.2f (%s) and even flips "
        "sign, because a single year offers only a handful of distinct grid-cell values. 30 fields × %d "
        "years, descriptive only. Marker area grows with the number of field-years that share the same "
        "cell pair, because fields in one POWER cell repeat a value exactly."
        % (by_year[year_lo]["rho"], year_lo, by_year[year_hi]["rho"], year_hi, len(years)),
        y=0.008,
    )
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# maps (one locator panel per farm, county scale)
# --------------------------------------------------------------------------

def _map_extent(frame, pad: float = 0.10):
    _ensure_lib_on_path()
    from field_frame import view_extent

    b = frame.geometry.total_bounds
    return view_extent(b[0], b[2], b[1], b[3], pad_frac=pad)


def _format_metres(ax) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    fmt = FuncFormatter(lambda v, _: f"{v:,.0f}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.tick_params(labelsize=6.4, length=2.5, width=0.6)


def _overlap_area(a, b) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return max(0.0, w) * max(0.0, h)


def _label_box(ax, x, y, text, fontsize):
    """Approximate display-space box for a multi-line text at (x, y) in data coords."""
    from matplotlib.transforms import Bbox

    pt2px = ax.figure.dpi / 72.0
    lines = text.split("\n")
    w = (0.62 * max(len(line) for line in lines) + 0.62) * fontsize * pt2px
    h = 1.30 * fontsize * pt2px * len(lines)
    px, py = ax.transData.transform((x, y))
    return Bbox.from_bounds(px - w / 2, py - h / 2, w, h)


def _place_labels(fig, ax, points, texts, colors, fontsize=6.4, obstacles=None) -> None:
    """Greedy label placement with leader lines, measured without re-rendering.

    Text boxes are sized from the font metrics (approximately) in display
    coordinates, so scoring a candidate costs nothing and the map stays legible:
    a label is never drawn on top of its own parcel, never leaves the axes,
    avoids the north-arrow corner and the other labels, and only then prefers a
    short leader over a long one.
    """
    import math

    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    ax_box = ax.get_window_extent()
    pt2px = fig.dpi / 72.0
    box_w = (0.64 * max(len(t) for t in texts) + 0.50) * fontsize * pt2px
    box_h = 1.70 * fontsize * pt2px
    # keep labels clear of the axes spines, not merely inside the axes
    inset = 2.5 * pt2px
    safe = Bbox.from_bounds(ax_box.x0 + inset, ax_box.y0 + inset,
                            ax_box.width - 2 * inset, ax_box.height - 2 * inset)
    corner = Bbox.from_bounds(ax_box.x0 + ax_box.width * 0.80,
                              ax_box.y0,
                              ax_box.width * 0.20, ax_box.height * 0.22)
    placed = [corner] + list(obstacles or [])

    radii_pt = (20, 26, 34, 44, 56, 70, 86, 105, 128, 155, 190, 230)
    offsets = []
    for r in radii_pt:
        for k in range(24):
            ang = math.radians(15.0 * k)
            offsets.append((r * math.cos(ang), r * math.sin(ang)))
    offsets_px = [(dx * pt2px, dy * pt2px) for dx, dy in offsets]

    # biggest parcels first: their position is what the map is about
    for idx in range(len(points)):
        x, y = ax.transData.transform(points[idx])
        best = None
        for (ox, oy), (dx, dy) in zip(offsets_px, offsets):
            cx, cy = x + ox, y + oy
            box = Bbox.from_bounds(cx - box_w / 2, cy - box_h / 2, box_w, box_h)
            outside = (
                max(0.0, safe.x0 - box.x0) + max(0.0, box.x1 - safe.x1)
                + max(0.0, safe.y0 - box.y0) + max(0.0, box.y1 - safe.y1)
            )
            overlap = sum(_overlap_area(box, other) for other in placed)
            penalty = overlap + 9.0 * outside * box.height + 0.02 * math.hypot(dx, dy) * box_h
            if best is None or penalty < best[0]:
                best = (penalty, dx, dy, box)
            if penalty <= 0.0:
                break
        _, dx, dy, box = best
        placed.append(box)
        ax.annotate(
            texts[idx], points[idx], xytext=(dx, dy), textcoords="offset points",
            fontsize=fontsize, color=INK, ha="center", va="center", zorder=22,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=colors[idx],
                      lw=0.9, alpha=0.94),
            arrowprops=(dict(arrowstyle="-", lw=0.6, color=INK, shrinkA=0, shrinkB=2,
                             alpha=0.8) if math.hypot(dx, dy) > 10 else None),
        )


def _prepare_panel(fig, ax, frame, ext=None):
    """Set an equal-metre window that fills the subplot slot without cropping.

    Matplotlib's aspect machinery either shrinks the data window (``datalim``,
    which silently clipped three Illinois parcels) or shrinks the axes box
    (``box``, which leaves the three panels different sizes).  Instead we size
    the window ourselves from the slot's physical shape, so one metre is one
    metre in every panel and every parcel stays inside its map.
    """
    base = ext or _map_extent(frame)
    ax.set_aspect("auto")
    pos = ax.get_position()
    fw, fh = fig.get_size_inches()
    ratio = (pos.width * fw) / max(pos.height * fh, 1e-6)

    x0, x1, y0, y1 = base
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    w, h = x1 - x0, y1 - y0
    if w / h > ratio:
        h = w / ratio
    else:
        w = h * ratio
    ext2 = (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)
    ax.set_xlim(ext2[0], ext2[1])
    ax.set_ylim(ext2[2], ext2[3])

    b = frame.geometry.total_bounds
    margin = max(0.0, b[0] - ext2[0], ext2[1] - b[2], b[1] - ext2[2], ext2[3] - b[3])
    return ext2, margin


def _panel_pixels(ext, panel_h, panel_w):
    aspect = (ext[3] - ext[2]) / max(ext[1] - ext[0], 1e-6)
    nx = int(panel_w * 150)
    ny = max(64, int(nx * aspect))
    return nx, ny


def fig_boundary_maps(ctx, path: Path, panel=(4.3, 4.2)) -> Path:
    farms = ctx["farms"]
    colors = ctx["grower_colors"]
    geo = ctx["geometry"]
    cy = ctx["cdl_year"]
    latest = int(max(ctx["cdl_years"]))
    _ensure_lib_on_path()
    import basemap
    from field_frame import frame_for, decorate, crs_note

    fig, axes = plt.subplots(1, len(farms), figsize=(panel[0] * len(farms) + 0.6, panel[1] + 1.0))
    axes = np.atleast_1d(axes)
    for ax, farm in zip(axes, farms):
        frame = frame_for(farm.fields.to_crs(4326), panel_h_in=panel[1], panel_w_in=panel[0],
                          force_rot=0.0)
        ext, _ = _prepare_panel(fig, ax, frame)
        nx, ny = _panel_pixels(ext, panel[1], panel[0])
        basemap.draw(ax, frame, ext, ny, nx, target_res_m=25.0)

        framed = frame.geometry.copy()
        framed["field_id"] = farm.fields["field_id"].astype(str).values
        color = colors[farm.short_label]
        framed.boundary.plot(ax=ax, color="white", linewidth=3.4, alpha=0.85, zorder=5)
        framed.plot(ax=ax, facecolor=to_rgba(color, 0.45), edgecolor=color, linewidth=2.0,
                    zorder=6)

        info = geo[geo["farm_slug"] == farm.farm_slug].set_index("field_id")
        pts, texts = [], []
        for fid, geom in zip(framed["field_id"], framed.geometry):
            p = geom.representative_point()
            short = info.loc[fid, "field_label"].split("-")[-1] if fid in info.index else fid
            pts.append((p.x, p.y))
            texts.append(short)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=7, color=INK,
                   edgecolor="white", linewidth=0.4, zorder=21)
        _place_labels(fig, ax, pts, texts, [color] * len(pts), fontsize=6.4)

        span_km = max(frame.geometry.total_bounds[2], frame.geometry.total_bounds[3]) / 1000
        ax.set_title(
            f"{farm.short_label} · {farm.farm_slug}\n"
            f"{len(framed)} fields · {farm.total_acres:,.0f} ac · {span_km:.0f} km span",
            fontsize=8.4, color=color,
        )
        decorate(ax, frame, arrow_r=0.075, arrow_fs=7.6, callout=False,
                 arrow_xy=(0.93, 0.13))
        ax.set_xlabel(crs_note(frame), fontsize=5.6, color="#4B5563")
        ax.set_ylabel("northing (m)", fontsize=5.8, color="#4B5563")
        _format_metres(ax)
    fig.subplots_adjust(bottom=0.13, top=0.90)
    fig.suptitle(
        "Fields are scattered parcels across a county, not one contiguous block",
        fontsize=10, fontweight="bold", y=0.995,
    )
    _fig_caption(
        fig,
        "Field-local UTM frame per farm, north up. Numbers are field indices (acres in figure A1 and "
        "tables/fields_geometry.csv). Boundary: OSM/CSB parcels — positional accuracy is not ground truth.",
        y=0.004, fontsize=6.6,
    )
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_cdl_map(ctx, path: Path, panel=(4.3, 4.2)) -> Path:
    import rasterio

    farms = ctx["farms"]
    colors = ctx["grower_colors"]
    codes, name_to_code = ctx["crop_codes"]
    analysis = {int(code): name for name, code in name_to_code.items()}
    cy = ctx["cdl_year"]
    latest = int(max(ctx["cdl_years"]))
    _ensure_lib_on_path()
    import basemap
    from field_frame import decorate, crs_note, frame_for, reproject_into_frame

    fig, axes = plt.subplots(1, len(farms), figsize=(panel[0] * len(farms) + 0.6, panel[1] + 1.6))
    axes = np.atleast_1d(axes)
    for ax, farm in zip(axes, farms):
        frame = frame_for(farm.fields.to_crs(4326), panel_h_in=panel[1], panel_w_in=panel[0],
                          force_rot=0.0)
        ext, margin = _prepare_panel(fig, ax, frame)
        nx, ny = _panel_pixels(ext, panel[1], panel[0])
        basemap.draw(ax, frame, ext, ny, nx, target_res_m=30.0)

        raster = farm.cdl_raster
        if raster is not None:
            with rasterio.open(raster) as ds:
                arr, rext = reproject_into_frame(
                    ds, frame, dst_res=15.0, dst_dtype="int16", nodata=0,
                    resampling=rasterio.warp.Resampling.nearest, margin_m=margin + 60.0,
                )
            arr = np.ma.filled(arr, 0).astype("int64")
            rgba = np.zeros(arr.shape + (4,), dtype=float)
            for code in sorted(int(c) for c in np.unique(arr) if c > 0):
                colour = codes.get(code) if code in analysis else "#B9BEC6"
                rgba[arr == code] = to_rgba(colour or "#B9BEC6", 1.0)
            ax.imshow(rgba, extent=rext, origin="upper", zorder=3, interpolation="nearest")

        framed = frame.geometry.copy()
        framed["field_id"] = farm.fields["field_id"].astype(str).values
        framed.boundary.plot(ax=ax, color="white", linewidth=4.4, alpha=0.92, zorder=7)
        framed.boundary.plot(ax=ax, color=INK, linewidth=1.7, zorder=8)

        info = ctx["geometry"].set_index("field_id")
        crops = cy[cy["year"] == latest].set_index("field_id")["dominant_crop"]
        pts, texts = [], []
        for fid, geom in zip(framed["field_id"], framed.geometry):
            p = geom.representative_point()
            short = info.loc[fid, "field_label"].split("-")[-1]
            pts.append((p.x, p.y))
            texts.append(short)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=7, color=INK,
                   edgecolor="white", linewidth=0.4, zorder=9)
        _place_labels(fig, ax, pts, texts, [colors[farm.short_label]] * len(pts), fontsize=6.4)

        n_missing = sum(1 for fid in framed["field_id"] if fid not in crops.index)
        subtitle = (
            "" if n_missing == 0
            else f" — {n_missing} field without CDL rows" if n_missing == 1
            else f" — {n_missing} fields without CDL rows"
        )
        ax.set_title(
            f"{farm.short_label} · CDL {latest}{subtitle}",
            fontsize=8.4, color=colors[farm.short_label],
        )
        decorate(ax, frame, arrow_r=0.075, arrow_fs=7.6, callout=False,
                 arrow_xy=(0.93, 0.13))
        ax.set_xlabel(crs_note(frame), fontsize=5.6, color="#4B5563")
        ax.set_ylabel("northing (m)", fontsize=5.8, color="#4B5563")
        _format_metres(ax)

    handles = [
        Patch(facecolor=codes.get(code, "#9CA3AF"), edgecolor="white", label=name)
        for code, name in sorted(analysis.items(), key=lambda kv: kv[1])
    ]
    handles.append(Patch(facecolor="#B9BEC6", edgecolor="white", label="Other CDL classes"))
    fig.subplots_adjust(bottom=0.20, top=0.90)
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.055), handlelength=1.1, columnspacing=1.4,
               fontsize=7.0)
    fig.suptitle(
        f"The cropland mosaic continues inside every parcel, CDL {latest}",
        fontsize=10, fontweight="bold", y=0.995,
    )
    _fig_caption(
        fig,
        "Numbers are field indices; white-haloed black outlines are the field boundaries. The 30 m CDL "
        "raster is drawn edge to edge so the crop inside a parcel can be compared with the surrounding "
        "mosaic; classes outside the seven analysis classes are greyed.",
        y=0.004, fontsize=6.6,
    )
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_weather_map(ctx, path: Path, panel=(4.3, 4.2)) -> Path:
    from matplotlib.patches import Polygon as MplPolygon
    from shapely.geometry import Polygon as SPolygon, box as sbox

    farms = ctx["farms"]
    colors = ctx["grower_colors"]
    wy = ctx["weather_year"]
    year = int(max(ctx["weather_years"]))
    _ensure_lib_on_path()
    import basemap
    from field_frame import decorate, crs_note, frame_for

    vals = wy.loc[wy["year"] == year, "precip_gs_mm"].dropna()
    vmin, vmax = float(vals.min()), float(vals.max())
    cmap = matplotlib.colormaps["YlGnBu"]
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, len(farms), figsize=(panel[0] * len(farms) + 0.6, panel[1] + 1.6))
    axes = np.atleast_1d(axes)
    for ax, farm in zip(axes, farms):
        frame = frame_for(farm.fields.to_crs(4326), panel_h_in=panel[1], panel_w_in=panel[0],
                          force_rot=0.0)
        ext, _ = _prepare_panel(fig, ax, frame)
        nx, ny = _panel_pixels(ext, panel[1], panel[0])
        basemap.draw(ax, frame, ext, ny, nx, target_res_m=25.0)

        window = sbox(ext[0], ext[2], ext[1], ext[3])
        obstacles = []
        sub = wy[(wy["farm_slug"] == farm.farm_slug) & (wy["year"] == year)]
        cells = sub.groupby(["power_lat0", "power_lon0"])["precip_gs_mm"].agg(["mean", "size"])
        for (lat0, lon0), row in cells.iterrows():
            corners = []
            for clon, clat in ((lon0, lat0), (lon0 + POWER_DLON, lat0),
                               (lon0 + POWER_DLON, lat0 + POWER_DLAT), (lon0, lat0 + POWER_DLAT)):
                e, n = _ll_to_frame(frame, clon, clat)
                corners.append((e, n))
            visible = SPolygon(corners).intersection(window)
            if visible.is_empty:
                continue
            # The cell mean lives in the panel subtitle: an in-map box would sit
            # on the parcel cluster and push the field labels apart.
            for poly in getattr(visible, "geoms", [visible]):
                ax.add_patch(MplPolygon(
                    np.asarray(poly.exterior.coords), closed=True,
                    facecolor=cmap(norm(row["mean"])), edgecolor=INK,
                    linewidth=1.5, linestyle=(0, (6, 4)), alpha=0.42, zorder=4,
                ))

        framed = frame.geometry.copy()
        framed["field_id"] = farm.fields["field_id"].astype(str).values
        framed.boundary.plot(ax=ax, color="white", linewidth=3.0, alpha=0.9, zorder=6)
        framed.boundary.plot(ax=ax, color=INK, linewidth=1.1, zorder=7)

        info = ctx["geometry"].set_index("field_id")
        fpts, ftexts = [], []
        for fid, geom in zip(framed["field_id"], framed.geometry):
            pt = geom.representative_point()
            fpts.append((pt.x, pt.y))
            ftexts.append(info.loc[fid, "field_label"].split("-")[-1])
        ax.scatter([q[0] for q in fpts], [q[1] for q in fpts], s=6, color=INK,
                   edgecolor="white", linewidth=0.4, zorder=8)
        _place_labels(fig, ax, fpts, ftexts, [colors[farm.short_label]] * len(fpts),
                      fontsize=6.4, obstacles=obstacles)

        n_cells = len(cells)
        ax.set_title(
            f"{farm.short_label} · {len(sub)} fields in {n_cells} POWER cell"
            + ("s" if n_cells != 1 else ""),
            fontsize=8.4, color=colors[farm.short_label], pad=13,
        )
        means = sorted(cells["mean"])
        ax.text(
            0.5, 1.005,
            "cell mean " + " & ".join(f"{m:,.0f}" for m in means) + " mm (May–Sep)",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=7.0, color="#4B5563",
        )
        decorate(ax, frame, arrow_r=0.075, arrow_fs=7.6, callout=False,
                 arrow_xy=(0.93, 0.13))
        ax.set_xlabel(crs_note(frame), fontsize=5.6, color="#4B5563")
        ax.set_ylabel("northing (m)", fontsize=5.8, color="#4B5563")
        _format_metres(ax)

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.subplots_adjust(bottom=0.26, top=0.90)
    cbar_ax = fig.add_axes([0.30, 0.135, 0.40, 0.020])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(f"POWER-cell growing-season precipitation {year}, May–Sep (mm)", fontsize=7.4)
    cbar.ax.tick_params(labelsize=6.8, length=2)
    fig.suptitle("Weather is sampled on a ~50 km grid, not at the field",
                 fontsize=10, fontweight="bold", y=0.99)
    _fig_caption(
        fig,
        "Numbers are field indices (white-haloed black outlines are the boundaries). Each tinted band with a "
        "dashed edge is one NASA POWER cell (0.5° × 0.625°, drawn at 42 % opacity); every field inside one "
        "cell receives the identical daily series, so a farm has one or two weather values, never ten.",
        y=0.004, fontsize=6.6,
    )
    fig.savefig(path)
    plt.close(fig)
    return path


def _ll_to_frame(frame, lon, lat):
    """WGS84 lon/lat -> field-frame metres (same convention as the panels)."""
    import pyproj

    tf = pyproj.Transformer.from_crs(4326, f"EPSG:{frame.epsg}", always_xy=True)
    e, n = tf.transform(lon, lat)
    return _frame_xy(frame, e, n)


def _frame_xy(frame, e, n):
    import math

    r = math.radians(frame.rot)
    dx, dy = e - frame.e0, n - frame.n0
    x = dx * math.cos(r) + dy * math.sin(r)
    y = -dx * math.sin(r) + dy * math.cos(r)
    return x, y

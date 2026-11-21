#!/usr/bin/env python3
"""Compare topsoil texture across demo farms from per-farm texture tables.

Reads the ``*_field_texture.csv`` files written by
``generate_field_texture_summary.py`` (one row per field: clay/silt/sand %,
USDA class, area), merges them, tests whether sand and clay differ between
growers, and renders one figure: the fields on the USDA triangle plus the
sand and clay distributions that separate them.

With three growers the tests are descriptive -- they say "these panels are
distinguishable", not "these populations differ".

Outputs, into ``--out-dir``:
    soil_texture_all_fields.csv   merged per-field table with a grower column
    soil_texture_comparison.png   triangle + sand/clay strips
    soil_texture_stats.json       Kruskal-Wallis, epsilon squared, Holm pairs

Usage::

    python generate_texture_comparison.py \
        --tables <il>/derived/tables/dekalb-demo-farm_field_texture.csv \
                 <ia>/derived/tables/story-demo-farm_field_texture.csv \
                 <ne>/derived/tables/york-demo-farm_field_texture.csv \
        --out-dir $DATA_PIPELINE_DATA_ROOT/data-pipeline/comparisons/soil-texture
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_SCRIPTS_DIR / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_soil_texture_triangles import (  # pyright: ignore[reportMissingImports]
    _CLASSES,
    _poly_verts,
    _to_xy,
)

GROWER_COLORS = {"IL": "#0072B2", "IA": "#D55E00", "NE": "#009E73"}
GROWER_NAMES = {"IL": "Illinois", "IA": "Iowa", "NE": "Nebraska"}
PARTS = ("clay_pct", "silt_pct", "sand_pct")


def load_tables(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if df.empty:
            continue
        df["grower"] = df["field_label"].str.split("-").str[0]
        df["farm"] = p.name.replace("_field_texture.csv", "")
        frames.append(df)
    if not frames:
        raise SystemExit("no rows read from the given texture tables")
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=list(PARTS)).sort_values(["grower", "field_label"]).reset_index(drop=True)


def holm(pvals: dict[tuple, float]) -> dict[tuple, float]:
    keys = sorted(pvals, key=lambda k: pvals[k])
    m = len(keys)
    adjusted, running = {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adjusted[k] = running
    return adjusted


def kruskal_block(df: pd.DataFrame, value: str, groups: list[str]) -> dict:
    from scipy import stats

    samples = [df.loc[df["grower"] == g, value].to_numpy() for g in groups]
    samples = [s for s in samples if len(s) > 1]
    H, p = stats.kruskal(*samples)
    n = sum(len(s) for s in samples)
    eps = max(0.0, (H - len(samples) + 1) / (n - len(samples))) if n > len(samples) else 0.0
    raw = {}
    for a, b in combinations(groups, 2):
        u, p_ab = stats.mannwhitneyu(
            df.loc[df["grower"] == a, value], df.loc[df["grower"] == b, value],
            alternative="two-sided",
        )
        raw[(a, b)] = float(p_ab)
    adj = holm(raw)
    return {
        "value": value,
        "test": "kruskal_wallis",
        "groups": {g: int((df["grower"] == g).sum()) for g in groups},
        "n": int(n),
        "H": round(float(H), 3),
        "p": round(float(p), 4),
        "epsilon_squared": round(float(eps), 3),
        "medians": {g: round(float(df.loc[df["grower"] == g, value].median()), 1) for g in groups},
        "pairwise_mannwhitney_holm": {
            f"{a}-{b}": {"p_raw": round(raw[(a, b)], 4), "p_holm": round(adj[(a, b)], 4)}
            for a, b in raw
        },
    }


def _strip_panel(ax, df: pd.DataFrame, stats: dict, value: str, growers: list[str],
                 rng, *, size: float = 34, title_size: float = 9.8) -> None:
    import numpy as np

    title = {"sand_pct": "Sand", "clay_pct": "Clay", "silt_pct": "Silt"}[value]
    for i, g in enumerate(growers):
        vals = df.loc[df["grower"] == g, value].to_numpy()
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=size,
                   color=GROWER_COLORS[g], edgecolor="white", linewidth=0.8, zorder=3)
        med = float(np.median(vals))
        ax.plot([i - 0.28, i + 0.28], [med, med], color="#111827", lw=1.6, zorder=4)
        ax.annotate(f"{med:.0f}", (i + 0.30, med), fontsize=7.5, color="#111827", va="center")
    block = stats[value]
    ax.set_xticks(range(len(growers)))
    ax.set_xticklabels([GROWER_NAMES[g] for g in growers], fontsize=8.5)
    ax.set_ylabel(f"{title} (%)", fontsize=8.5)
    ax.tick_params(axis="y", labelsize=7.6)
    p_txt = "< 0.001" if block["p"] < 0.001 else f"= {block['p']:.3f}"
    ax.set_title(f"{title} by grower  ·  KW p {p_txt}, \u03b5\u00b2 = {block['epsilon_squared']:.2f}",
                 fontsize=title_size, loc="left", pad=6)
    ax.grid(axis="y", color="#e5e7eb", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.margins(y=0.18)


def render_panel(df: pd.DataFrame, stats: dict, value: str, target: Path) -> None:
    """One component on its own, for use as a standalone report figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    growers = [g for g in ("IL", "IA", "NE") if g in set(df["grower"])]
    fig, ax = plt.subplots(figsize=(6.4, 4.1), dpi=200)
    _strip_panel(ax, df, stats, value, growers, np.random.default_rng(20261121),
                 size=52, title_size=10.5)
    fig.text(0.012, 0.012,
             "Each point is one field (0-20 cm, SSURGO area-weighted); bar = grower median.",
             fontsize=7, color="#4b5563", ha="left")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.18)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render(df: pd.DataFrame, stats: dict, target: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Polygon as MplPolygon

    growers = [g for g in ("IL", "IA", "NE") if g in set(df["grower"])]
    fig = plt.figure(figsize=(13.2, 6.6), dpi=200)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1, 1],
                          wspace=0.22, hspace=0.42)
    ax_t = fig.add_subplot(gs[:, 0])
    ax_sand = fig.add_subplot(gs[0, 1])
    ax_clay = fig.add_subplot(gs[1, 1])

    # --- triangle ----------------------------------------------------------
    # A full triangle spends most of its area on clay-rich classes none of these
    # fields occupy, but cropping a ternary plot to the data band would distort
    # the axes, so the reference triangle stays whole. Fields cluster tightly,
    # so growers get distinct marker shapes on top of colour: overlapping IL and
    # NE points stay countable.
    for name, (idx, color) in _CLASSES.items():
        ax_t.add_patch(MplPolygon(_poly_verts(idx), closed=True, facecolor=color,
                                  edgecolor="white", lw=0.8, alpha=0.9, zorder=1))
    ax_t.add_patch(MplPolygon([_to_xy(0, 1, 0), _to_xy(0, 0, 1), _to_xy(1, 0, 0)],
                              closed=True, fill=False, edgecolor="#5b4a3f", lw=1.3, zorder=3))
    for name, (idx, _) in _CLASSES.items():
        crowded = name in {"silty loam", "loam"}
        if crowded:  # the two cells the points land in; labelled in the caption
            continue
        cx, cy = _poly_verts(idx).mean(axis=0)
        ax_t.text(cx, cy, name, ha="center", va="center", fontsize=5.4,
                  color="#4b3a34", alpha=0.95, zorder=2)
    markers = {"IL": "o", "IA": "s", "NE": "^"}
    for g in growers:
        sub = df[df["grower"] == g]
        xs, ys = zip(*(_to_xy(c / 100, s / 100, sa / 100)
                       for c, s, sa in zip(sub["clay_pct"], sub["silt_pct"], sub["sand_pct"])))
        ax_t.scatter(xs, ys, s=52, marker=markers[g], color=GROWER_COLORS[g],
                     edgecolor="white", linewidth=0.9, alpha=0.95, zorder=6,
                     label=f"{GROWER_NAMES[g]} ({len(sub)})")
    ax_t.set_xlim(-0.03, 1.05)
    ax_t.set_ylim(-0.05, 0.90)
    ax_t.axis("off")
    ax_t.set_title("Topsoil texture, 0-20 cm", fontsize=11.5, loc="left", pad=8)
    ax_t.text(0.50, -0.035, "Sand (%)", ha="center", fontsize=8.5)
    ax_t.text(0.10, 0.46, "Clay (%)", va="center", rotation=60, fontsize=8.5)
    ax_t.text(0.86, 0.46, "Silt (%)", va="center", rotation=-60, fontsize=8.5)
    ax_t.legend(loc="upper right", fontsize=8, frameon=False, handletextpad=0.4)

    # --- sand / clay strips -----------------------------------------------
    rng = np.random.default_rng(20261121)
    for ax, value in ((ax_sand, "sand_pct"), (ax_clay, "clay_pct")):
        _strip_panel(ax, df, stats, value, growers, rng, size=34, title_size=9.8)

    fig.suptitle("Soil texture differences across the three demo growers",
                 fontsize=13, x=0.012, ha="left")
    fig.text(0.012, 0.012,
             "Each point is one field (0-20 cm, SSURGO area-weighted); the two occupied triangle cells are "
             "silt loam and loam. Horizontal bar = grower median. Three growers: descriptive.",
             fontsize=7.4, color="#4b5563", ha="left")
    fig.subplots_adjust(left=0.03, right=0.98, top=0.90, bottom=0.06)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", nargs="+", required=True, type=Path,
                    help="per-farm *_field_texture.csv files")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(os.environ.get("DATA_PIPELINE_DATA_ROOT", "."))
                    / "data-pipeline" / "comparisons" / "soil-texture")
    args = ap.parse_args(argv)

    df = load_tables([Path(p) for p in args.tables])
    growers = [g for g in ("IL", "IA", "NE") if g in set(df["grower"])]
    stats = {v: kruskal_block(df, v, growers) for v in ("sand_pct", "clay_pct", "silt_pct")}

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "soil_texture_all_fields.csv", index=False)
    render(df, stats, out / "soil_texture_comparison.png")
    render_panel(df, stats, "sand_pct", out / "soil_texture_sand.png")
    render_panel(df, stats, "clay_pct", out / "soil_texture_clay.png")
    payload = {
        "generated_by": "generate_texture_comparison.py",
        "fields": int(len(df)),
        "growers": growers,
        "depth_basis": "0-20 cm, component- and map-unit-area weighted (SSURGO)",
        "tests": stats,
        "classes_by_grower": {
            g: df.loc[df["grower"] == g, "texture_class"].value_counts().to_dict() for g in growers
        },
    }
    (out / "soil_texture_stats.json").write_text(json.dumps(payload, indent=2) + "\n")

    for v, block in stats.items():
        med = ", ".join(f"{g} {m:g}" for g, m in block["medians"].items())
        print(f"{v:<9} KW H = {block['H']:.2f}  p = {block['p']:.3f}  "
              f"e2 = {block['epsilon_squared']:.2f}  medians: {med}")
    print(f"\n{len(df)} fields across {len(growers)} growers -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

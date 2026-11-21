#!/usr/bin/env python3
"""One row of topsoil texture per field, area- and component-weighted from SSURGO.

Basis: the ``0-20 cm`` plow layer. Each soil component's texture is the
thickness-weighted mean of the SSURGO horizons overlapping that interval;
components are weighted by ``comppct_r`` inside their map unit, and map units
by their clipped polygon area in ``soil/ssurgo_soil_types.geojson``. The USDA
class comes from the same triangle geometry the texture-triangle figures use
(``generate_soil_texture_triangles``), so the table and the figures can never
disagree about a field's class.

Inputs, per field, all written by ``ingest/download_soil.py`` from the NRCS
Soil Data Access service that backs Web Soil Survey::

    <farm>/fields/<field_id>/soil/ssurgo_full.csv          long horizon table
    <farm>/fields/<field_id>/soil/ssurgo_soil_types.geojson map unit polygons

Output: ``<farm>/derived/tables/<farm>_field_texture.csv`` plus a farm-level
texture triangle at ``<farm>/derived/features/<farm>_soil_texture_triangle.png``.

Usage::

    python generate_field_texture_summary.py --farm-dir <farm> [--depth 20] \
        [--no-figure]

Notes:
    * Sand/silt/clay are reported normalised to 100 %, because SSURGO particle
      size totals are occasionally 96-99 %; the pre-normalisation total is kept
      as ``psd_total_pct`` so the adjustment stays visible.
    * Fields whose components carry no particle-size data are reported with
      blank texture columns rather than an invented class.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_SCRIPTS_DIR / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# One source of truth for the USDA triangle: the class polygons are defined in
# the texture-triangle renderer, so reuse them instead of re-deriving thresholds.
from generate_soil_texture_triangles import (  # pyright: ignore[reportMissingImports]
    _CLASSES,
    _poly_verts,
    _to_xy,
    texture_class,
)

PSD_COLUMNS = ("claytotal_r", "silttotal_r", "sandtotal_r")


def field_dirs(farm_dir: Path) -> list[Path]:
    return sorted(p for p in (farm_dir / "fields").iterdir() if p.is_dir())


def field_label_map(farm_dir: Path) -> dict[str, str]:
    """Field labels by descending boundary area, matching the EDA subskill.

    ``NE-01`` is the largest NE field, ``NE-10`` the smallest.
    """
    import geopandas as gpd

    rows = []
    for d in field_dirs(farm_dir):
        bnd = d / "boundary" / "field_boundary.geojson"
        if not bnd.exists():
            continue
        g = gpd.read_file(bnd)
        rows.append((d.name, float(g.to_crs(6933).geometry.area.sum() / 4046.8564224)))
    rows.sort(key=lambda r: -r[1])
    return {fid: f"{i:02d}" for i, (fid, _) in enumerate(rows, start=1)}


def polygon_areas(geojson: Path) -> dict[str, float]:
    """Map unit area in m^2, summed per mukey (polygons are already field-clipped)."""
    import geopandas as gpd

    g = gpd.read_file(geojson)
    if g.empty:
        return {}
    g = g.to_crs(6933)
    g["mukey"] = g["mukey"].astype(str)
    return g.assign(a=g.geometry.area).groupby("mukey")["a"].sum().to_dict()


def component_texture(soil: pd.DataFrame, depth_cm: float) -> pd.DataFrame:
    """Thickness-weighted 0-``depth_cm`` texture per component.

    A component contributes only the part of each horizon that falls inside the
    interval, so a 0-16 cm horizon counts fully and a 16-40 cm horizon only
    contributes its first 4 cm.
    """
    df = soil.copy()
    df["mukey"] = df["mukey"].astype(str)
    df = df.dropna(subset=list(PSD_COLUMNS) + ["hzdept_r", "hzdepb_r"])

    top = df["hzdept_r"].clip(lower=0.0)
    bottom = df["hzdepb_r"].clip(upper=depth_cm)
    df["overlap"] = (bottom - top).clip(lower=0.0)
    df = df[df["overlap"] > 0]
    df = df[df["comppct_r"].fillna(0) > 0]
    if df.empty:
        return df

    grouped = df.groupby(["mukey", "compname"], as_index=False)
    out = grouped.apply(
        lambda g: pd.Series(
            {
                "comppct_r": float(g["comppct_r"].iloc[0]),
                "clay": (g["claytotal_r"] * g["overlap"]).sum() / g["overlap"].sum(),
                "silt": (g["silttotal_r"] * g["overlap"]).sum() / g["overlap"].sum(),
                "sand": (g["sandtotal_r"] * g["overlap"]).sum() / g["overlap"].sum(),
                "depth_cm": float(g["overlap"].sum()),
            }
        ),
        include_groups=False,
    )
    return out.reset_index(drop=True)


def field_texture(field_dir: Path, depth_cm: float) -> dict | None:
    soil_csv = field_dir / "soil" / "ssurgo_full.csv"
    if not soil_csv.exists():
        return None
    soil = pd.read_csv(soil_csv)
    if soil.empty:
        return None

    comps = component_texture(soil, depth_cm)
    if comps.empty:
        return {
            "field_id": field_dir.name,
            "map_units": int(soil["mukey"].nunique()),
            "psd_total_pct": None,
            "clay_pct": None,
            "silt_pct": None,
            "sand_pct": None,
            "texture_class": None,
        }

    # component weight = share of its map unit x map unit area
    comps["share_in_mu"] = comps["comppct_r"] / comps.groupby("mukey")["comppct_r"].transform("sum")
    areas = polygon_areas(field_dir / "soil" / "ssurgo_soil_types.geojson")
    comps["mu_weight"] = comps["mukey"].map(areas)
    if comps["mu_weight"].isna().any():
        # no polygon table for some mukeys: fall back to equal map unit weights
        comps["mu_weight"] = comps["mu_weight"].fillna(comps["mu_weight"].dropna().median() if comps["mu_weight"].notna().any() else 1.0)
    comps["weight"] = comps["mu_weight"] * comps["share_in_mu"]

    raw = {c: float((comps[c] * comps["weight"]).sum() / comps["weight"].sum())
           for c in ("clay", "silt", "sand")}
    total = sum(raw.values())
    clay, silt, sand = (100.0 * raw[c] / total for c in ("clay", "silt", "sand"))
    cls = texture_class(clay / 100.0, silt / 100.0, sand / 100.0)

    series_share = comps.groupby("compname")["weight"].sum()
    series_share = 100.0 * series_share / series_share.sum()
    dominant = series_share.idxmax()

    return {
        "field_id": field_dir.name,
        "area_acres": round(sum(areas.values()) / 4046.8564224, 2) if areas else None,
        "map_units": int(comps["mukey"].nunique()),
        "components": int(len(comps)),
        "dominant_series": dominant,
        "dominant_series_pct": round(float(series_share.max()), 1),
        "clay_pct": round(clay, 1),
        "silt_pct": round(silt, 1),
        "sand_pct": round(sand, 1),
        "psd_total_pct": round(total, 1),
        "texture_class": cls,
        "topsoil_cm": depth_cm,
    }


def farm_state(farm_dir: Path) -> str:
    """Two-letter state for field labels: farm.json first, grower dir name second."""
    meta = farm_dir / "farm.json"
    if meta.exists():
        import json

        state = json.loads(meta.read_text()).get("state")
        if state:
            return str(state).upper()
    return farm_dir.parent.parent.name.split("-")[0].upper()


def build_table(farm_dir: Path, depth_cm: float) -> pd.DataFrame:
    labels = field_label_map(farm_dir)
    state = farm_state(farm_dir)
    rows = []
    for d in field_dirs(farm_dir):
        label = f"{state}-{labels.get(d.name, '??')}"
        row = field_texture(d, depth_cm)
        if row is None:
            print(f"  skip  {label} ({d.name}): no SSURGO table")
            continue
        row["field_label"] = label
        row["soil_source"] = "ssurgo"
        row["depth_basis"] = f"0-{depth_cm:g} cm thickness-weighted"
        rows.append(row)
    if not rows:
        raise SystemExit(f"no SSURGO soil tables under {farm_dir / 'fields'}")
    cols = [
        "field_label", "field_id", "area_acres", "map_units", "components",
        "dominant_series", "dominant_series_pct", "clay_pct", "silt_pct",
        "sand_pct", "psd_total_pct", "texture_class", "topsoil_cm",
        "depth_basis", "soil_source",
    ]
    return pd.DataFrame(rows)[cols].sort_values("field_label").reset_index(drop=True)


def render_figure(table: pd.DataFrame, target: Path, farm_name: str) -> None:
    """Two panels: where the fields sit on the USDA triangle, and their mixes.

    All ten NE fields land inside one class region, so a single labelled
    triangle would stack ten labels on the same spot. The triangle therefore
    carries unlabelled points and the per-field identity moves to a stacked
    composition panel where every label has its own row.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    pts = table.dropna(subset=["clay_pct", "silt_pct", "sand_pct"]).copy()
    pts = pts.sort_values("sand_pct", ascending=False)
    classes = pts["texture_class"].replace({None: "unclassified"})

    fig, (ax_t, ax_b) = plt.subplots(
        1, 2, figsize=(12.4, 6.4), dpi=200, gridspec_kw={"width_ratios": [1.0, 1.15]}
    )

    # --- panel A: USDA triangle -------------------------------------------
    for name, (idx, color) in _CLASSES.items():
        ax_t.add_patch(MplPolygon(_poly_verts(idx), closed=True, facecolor=color,
                                  edgecolor="white", lw=0.9, zorder=1))
    ax_t.add_patch(MplPolygon([_to_xy(0, 1, 0), _to_xy(0, 0, 1), _to_xy(1, 0, 0)],
                              closed=True, fill=False, edgecolor="#5b4a3f", lw=1.3, zorder=3))
    xs, ys = zip(*(_to_xy(c / 100, s / 100, sa / 100) for c, s, sa in
                   zip(pts["clay_pct"], pts["silt_pct"], pts["sand_pct"])))
    ax_t.plot(xs, ys, "o", ms=9, mfc="#1d4ed8", mec="white", mew=1.3, zorder=6)
    # zoom frame around the cluster, mirrored in the right panel
    pad = 0.045
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad * 0.8, max(ys) + pad * 0.8
    ax_t.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                 edgecolor="#1d4ed8", lw=1.0, ls=(0, (3, 2)), zorder=7))
    ax_t.text(x1 + 0.01, y1, "detail →", fontsize=7, color="#1d4ed8", va="center", zorder=7)
    # class names last, so any cell the frame or the dots sit on is skipped
    # instead of being printed underneath them
    for name, (idx, _) in _CLASSES.items():
        cx, cy = _poly_verts(idx).mean(axis=0)
        covered = (x0 - 0.09 < cx < x1 + 0.12) and (y0 - 0.07 < cy < y1 + 0.07)
        if covered:
            continue
        ax_t.text(cx, cy, name, ha="center", va="center", fontsize=5.6,
                  color="#4b3a34", zorder=2)
    ax_t.set_xlim(-0.03, 1.05)
    ax_t.set_ylim(-0.05, 0.90)
    ax_t.axis("off")
    ax_t.set_title("USDA texture triangle", fontsize=11, pad=10, loc="left")
    ax_t.text(0.50, -0.035, "Sand (%)", ha="center", fontsize=8.5)
    ax_t.text(0.10, 0.46, "Clay (%)", va="center", rotation=60, fontsize=8.5)
    ax_t.text(0.86, 0.46, "Silt (%)", va="center", rotation=-60, fontsize=8.5)

    # --- panel B: per-field composition (detail) --------------------------
    y = range(len(pts))
    height = 0.62
    colours = {"silt": "#b7c9a8", "clay": "#a8534f", "sand": "#e2c98a"}
    left = [0.0] * len(pts)
    for part, col in (("silt_pct", "silt"), ("clay_pct", "clay"), ("sand_pct", "sand")):
        vals = list(pts[part])
        ax_b.barh(y, vals, left=left, height=height, color=colours[col],
                  edgecolor="white", lw=0.8, label=col.capitalize())
        for yi, v, l in zip(y, vals, left):
            if v >= 9:
                ax_b.text(l + v / 2, yi, f"{v:.0f}", ha="center", va="center",
                          fontsize=7.2, color="#1f2937")
        left = [a + b for a, b in zip(left, vals)]
    ax_b.set_yticks(list(y))
    ax_b.set_yticklabels(pts["field_label"], fontsize=8.5)
    ax_b.set_xlim(0, 120)
    ax_b.set_xlabel("share of particle-size total (%)", fontsize=8.5)
    ax_b.tick_params(axis="x", labelsize=8)
    ax_b.set_title("Detail: topsoil composition, 0-20 cm", fontsize=11, pad=22, loc="left")
    ax_b.grid(axis="x", color="#e5e7eb", lw=0.7, zorder=0)
    ax_b.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_b.spines[spine].set_visible(False)
    for yi, (cls, acres, series) in enumerate(zip(classes, pts["area_acres"], pts["dominant_series"])):
        ax_b.text(101, yi, f"{cls} · {series} · {acres:.1f} ac", fontsize=6.8,
                  color="#4b5563", va="center")
    ax_b.legend(loc="lower left", fontsize=7.6, frameon=False, ncol=3,
                bbox_to_anchor=(0.0, 1.02), handlelength=1.1, columnspacing=1.6)

    fig.suptitle(f"Topsoil texture by field - {farm_name}, 0-20 cm, SSURGO area-weighted",
                 fontsize=12.5, x=0.02, ha="left")
    fig.text(0.02, 0.015,
             "Each field is the component- and map-unit-area-weighted mean of its SSURGO "
             "horizons in 0-20 cm; classes from the USDA triangle. The dashed frame marks "
             "where this farm's fields cluster.",
             fontsize=7.2, color="#4b5563", ha="left")
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--farm-dir", required=True, type=Path,
                    help="farm root, e.g. <runtime>/growers/ne-york-grower/farms/york-demo-farm")
    ap.add_argument("--depth", type=float, default=20.0, help="topsoil depth in cm (default 20)")
    ap.add_argument("--out", type=Path, help="CSV path (default <farm>/derived/tables/<farm>_field_texture.csv)")
    ap.add_argument("--no-figure", action="store_true", help="skip the texture triangle")
    args = ap.parse_args(argv)

    farm_dir = args.farm_dir.resolve()
    table = build_table(farm_dir, args.depth)
    out = args.out or farm_dir / "derived" / "tables" / f"{farm_dir.name}_field_texture.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    missing = table[table["texture_class"].isna()]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table[["field_label", "area_acres", "dominant_series", "clay_pct",
                     "silt_pct", "sand_pct", "texture_class"]].to_string(index=False))
    print(f"\n{len(table)} field(s), {len(table) - len(missing)} with texture; wrote {out}")
    if len(missing):
        print(f"  no particle-size data: {', '.join(missing['field_label'])}")

    if not args.no_figure:
        fig_path = farm_dir / "derived" / "features" / f"{farm_dir.name}_soil_texture_triangle.png"
        render_figure(table, fig_path, farm_dir.name.replace("-", " "))
        print(f"  figure: {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

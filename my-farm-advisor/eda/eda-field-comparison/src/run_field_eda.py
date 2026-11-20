#!/usr/bin/env python3
"""Field-level EDA across all grower farms in the runtime pipeline.

Reads boundaries, CDL composition, and daily weather already produced by the
data pipeline, then writes static figures plus tidy tables and a machine
readable manifest. No report prose, no HTML, no dashboard.

    python src/run_field_eda.py --output-dir "$DATA_PIPELINE_DATA_ROOT/data-pipeline/eda/field-eda"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eda_data as data
import eda_figures as figures
import eda_stats as stats


# Comparison levels every figure declares. Keeping the vocabulary closed makes
# the coverage auditable from manifest.json instead of prose.
LEVELS = {
    "field": "one row per field: cross-sectional geometry or five-year means",
    "field-year": "one row per field per year: repeated observations and change over time",
    "grower": "grouped by grower: differences among the fields of one grower",
    "across-grower": "growers compared with each other",
}


FIGURES = [
    dict(
        id="A1", key="area_by_grower", file="01_field_area_by_grower.png",
        title="Field size distribution by grower",
        question="How different are the operating units across the three growers?",
        story="IL runs ~6-14x larger parcels than IA/NE (median 98 vs 7 vs 15 ac); total acres "
              "span 2,354 / 489 / 329, so the same 10-field sample covers very different ground.",
        levels=["field", "grower", "across-grower"],
        caveat="10 fields per grower; the Kruskal-Wallis p-value is descriptive, not a "
               "population test.",
    ),
    dict(
        id="A2", key="area_vs_compactness", file="02_area_vs_compactness.png",
        title="Field area vs shape compactness",
        question="Are small fields also awkward shapes that remote sensing handles badly?",
        story="Compactness falls with area (ρ = -0.48): the 250-730 ac parcels are long or "
              "perforated, while sub-10-acre parcels are simple rectangles. Shape is not sensing "
              "quality — sub-acre fields are weak because they are only a few 30 m pixels wide.",
        levels=["field", "grower", "across-grower"],
        caveat="Static geometry only — shape has no time dimension. PP is scale-free but "
               "sensitive to digitising noise on sub-acre parcels.",
    ),
    dict(
        id="A3", key="cdl_coverage", file="03_cdl_coverage.png",
        title="CDL cover vs field area, and its within-field stability",
        question="Does the CDL classify the whole field, and does that change year to year?",
        story="25 of 29 fields classify at 95 %+; the four gaps are the 0.6-17 ac parcels where "
              "mixed edge pixels dominate. Each field's cover moves <2 pp across five years, so a "
              "gap is an edge effect, not annual change.",
        levels=["field", "field-year", "grower", "across-grower"],
        caveat="CDL is 30 m; sub-10-acre parcels can lose a large share of pixels to mixed edges.",
    ),
    dict(
        id="A-map", key="boundary_maps", file="04_farm_boundary_maps.png",
        title="Farm boundary locator maps",
        question="How are the parcels arranged on the ground?",
        story="Each 'farm' is 10 parcels scattered across a county (20-54 km span), not one "
              "contiguous block; scale and dispersion differ sharply between growers.",
        levels=["field", "grower", "across-grower"],
        caveat="OSM/CSB boundaries are not survey-grade. County-scale panels show layout, not "
               "within-field detail.",
    ),
    dict(
        id="B1", key="cdl_composition", file="05_cdl_composition_by_field.png",
        title="Mean crop composition per field",
        question="What is each field made of, 2019-2023?",
        story="Corn/soy dominate the arable parcels, but whole parcels are forest/grass in IA and "
              "NE; IL also carries wheat and alfalfa. A mean composition hides rotation.",
        levels=["field", "grower", "across-grower"],
        caveat="Shares are averaged over five years and renormalised to 100 %, hiding both "
               "rotation and coverage loss.",
    ),
    dict(
        id="B2", key="cdl_rotation", file="06_cdl_rotation_heatmap.png",
        title="Dominant CDL class by field and year",
        question="Which fields rotate, which are continuous, which are not cropland at all?",
        story="Continuous corn, clean corn-soy alternation, and stable forest/grass all appear; "
              "IA has one field with no CDL rows at all (blank, not zero).",
        levels=["field", "field-year", "grower", "across-grower"],
        caveat="Dominant class only; a 60/40 split is drawn as one colour. Mixed pixels can flip "
               "the dominant class on small parcels.",
    ),
    dict(
        id="B3", key="rotation_metrics", file="07_rotation_change_and_diversity.png",
        title="Rotation change count and crop diversity",
        question="How stable is the crop sequence, and does parcel size explain the variation?",
        story="Rotation is field-specific, not grower-specific: blocks alternate reliably or stay "
              "continuous within the same farm (KW p = 0.16, ε² = 0.06), and diversity has no "
              "size trend (ρ = -0.10, p = 0.59).",
        levels=["field", "field-year", "grower", "across-grower"],
        caveat="Four transitions per field is a tiny sample; class-change counts are sensitive to "
               "CDL classification noise.",
    ),
    dict(
        id="B-map", key="cdl_map", file="08_cdl_2023_map.png",
        title="CDL cropland map with field outlines",
        question="How do the fields sit inside the surrounding cropland mosaic?",
        story="Arable parcels are embedded in a mixed county mosaic; the NE/IA farm includes "
              "forest and pasture parcels that are not cropland at all.",
        levels=["field", "grower", "across-grower"],
        caveat="One CDL year only. Raster shown unmasked so surrounding land use stays visible.",
    ),
    dict(
        id="C1", key="season_precip", file="09_growing_season_precip.png",
        title="Growing-season precipitation by grower and year",
        question="What dominates variation in seasonal water supply?",
        story="Year-to-year swings dominate: a grower's median season moves 144-288 mm between "
              "its driest and wettest year, while grower medians differ by ~50 mm (IL highest, "
              "NE lowest). Fields inside one grower move together because they share a POWER cell.",
        levels=["field-year", "grower", "across-grower"],
        caveat="NASA POWER ~50 km cells; IA's 10 fields share ONE daily series, so field spread "
               "inside a grower is not microclimate.",
    ),
    dict(
        id="C2", key="cumulative_gdd", file="10_cumulative_gdd_2025.png",
        title="Cumulative growing-degree days, 2025",
        question="Does heat accumulation differ inside a farm or only between farms?",
        story="Curves inside a farm overlap exactly — one or two NASA POWER cells per farm, never "
              "one per field — while growers separate, with NE accumulating the most heat.",
        levels=["field-year", "grower"],
        caveat="Base 10 °C, May-Sep window, no crop-specific base temperature; overlap is grid "
               "resolution, not identical fields.",
    ),
    dict(
        id="C3", key="weather_variance", file="11_weather_variance_and_scatter.png",
        title="Weather variance decomposition and precip-GDD relationship",
        question="How much of the weather signal is year, grower, or field?",
        story="Rain is year-driven (58 %) and heat is grower-driven (81 %); field-within-grower is "
              "0-2 % in both. The pooled precip-GDD ρ = -0.54 is a between-year relation — "
              "inside a single year it runs from -0.69 to +0.32.",
        levels=["field", "field-year", "grower", "across-grower"],
        caveat="Nested sequential R² on 30 fields x 5 years; descriptive shares, no causal claim.",
    ),
    dict(
        id="C-map", key="weather_map", file="12_weather_grid_cells.png",
        title="Weather sampling grid vs field footprints",
        question="Why do fields inside one farm report identical weather?",
        story="Fields fall into one or two NASA POWER cells per farm; IL and NE straddle a cell "
              "boundary, IA sits inside a single cell.",
        levels=["field", "grower", "across-grower"],
        caveat="Cell registration follows the POWER 0.5 deg x 0.625 deg lattice with centres at "
               "0.25 + 0.5k; verified against identical-series groups in all three farms.",
    ),
]


def level_coverage() -> dict:
    """Figure ids that exercise each comparison level."""
    used = {lvl for spec in FIGURES for lvl in spec["levels"]}
    unknown = sorted(used - set(LEVELS))
    if unknown:
        raise SystemExit(f"unknown comparison level(s): {', '.join(unknown)}")
    return {lvl: [spec["id"] for spec in FIGURES if lvl in spec["levels"]] for lvl in LEVELS}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where figures/, tables/, comparisons.json and manifest.json go. "
                        "Defaults to <DATA_PIPELINE_DATA_ROOT>/data-pipeline/eda/field-eda")
    p.add_argument("--growers", default=None,
                   help="Comma-separated grower slugs to include (default: all complete farms)")
    p.add_argument("--expected-fields", type=int, default=10,
                   help="Input gate: required field count per farm (0 disables)")
    p.add_argument("--cdl-years", default="2019-2023")
    p.add_argument("--weather-years", default="2021-2025")
    p.add_argument("--figures", default="all",
                   help="Comma-separated figure ids to render (default: all)")
    p.add_argument("--skip-maps", action="store_true", help="Skip the three map figures")
    p.add_argument("--list-figures", action="store_true")
    return p.parse_args(argv)


def parse_years(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_figures:
        for fig in FIGURES:
            print(f"{fig['id']:>5}  {fig['file']:<34} {fig['title']}")
        return 0

    figures.setup_style()
    cdl_years = parse_years(args.cdl_years)
    weather_years = parse_years(args.weather_years)
    pipeline_root = data.runtime_pipeline_root()
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = pipeline_root / "eda" / "field-eda"
    out_dir = out_dir.expanduser().resolve()
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    growers = [g for g in (args.growers or "").split(",") if g.strip()] or None
    farms = data.discover_farms(pipeline_root, growers)
    farms.sort(key=lambda f: f.total_acres, reverse=True)

    gate = data.validate_inputs(
        farms,
        expected_fields=args.expected_fields or None,
        cdl_years=cdl_years,
        weather_years=weather_years,
    )

    geometry = data.fields_geometry_table(farms)
    weather_year = data.weather_field_year(farms, geometry)
    cdl_year = data.cdl_field_year(farms, geometry)
    composition = data.cdl_composition_mean(farms, geometry)
    rotation = data.rotation_metrics(geometry, cdl_year)
    cdl_matrix = data.cdl_matrix(farms, geometry, cdl_years)
    series_counts = data.weather_series_counts(farms)

    tables = {
        "fields_geometry.csv": geometry,
        "cdl_field_year.csv": cdl_year,
        "cdl_composition_mean.csv": composition,
        "field_rotation.csv": rotation,
        "cdl_matrix.csv": cdl_matrix,
        "weather_field_year.csv": weather_year,
    }
    for name, df in tables.items():
        df.to_csv(table_dir / name, index=False)

    grower_order = list(dict.fromkeys(geometry["grower_label"]))
    colors = figures.grower_colors(grower_order)
    crop_codes, name_to_code = figures.crop_palette(farms)

    # ---------------------------------------------------------------- stats
    latest = max(cdl_years)
    coverage_latest = cdl_year[cdl_year["year"] == latest].merge(
        geometry[["field_id", "area_acres"]], on="field_id", how="left"
    )
    stats_out: dict = {}
    stats_out["area_grower_kw"] = stats.kruskal_epsilon(geometry, "area_acres", "grower_label")
    stats_out["area_grower_pairwise"] = stats.pairwise_mannwhitney(
        geometry, "area_acres", "grower_label"
    )
    stats_out["area_descriptives"] = stats.describe(geometry, "area_acres", "grower_label")
    stats_out["area_compactness_spearman"] = stats.spearman_perm(
        geometry.assign(log_area=np.log10(geometry["area_acres"])),
        "log_area", "compactness_pp",
    )
    stats_out["cdl_coverage_area_spearman"] = stats.spearman_perm(
        coverage_latest.assign(log_area=np.log10(coverage_latest["area_acres"])),
        "log_area", "cdl_coverage_pct",
    )
    stats_out["rotation_changes_kw"] = stats.kruskal_epsilon(
        rotation, "crop_changes", "grower_label"
    )
    stats_out["rotation_changes_pairwise"] = stats.pairwise_mannwhitney(
        rotation, "crop_changes", "grower_label"
    )
    stats_out["diversity_area_spearman"] = stats.spearman_perm(
        rotation.assign(log_area=np.log10(rotation["area_acres"])),
        "log_area", "crop_diversity_shannon",
    )
    stats_out["precip_grower_kw"] = stats.kruskal_epsilon(
        weather_year, "precip_gs_mm", "grower_label"
    )
    stats_out["precip_grower_pairwise"] = stats.pairwise_mannwhitney(
        weather_year, "precip_gs_mm", "grower_label"
    )
    stats_out["precip_descriptives"] = stats.describe(
        weather_year, "precip_gs_mm", "grower_label"
    )
    stats_out["precip_decomp"] = stats.variance_decomposition(weather_year, "precip_gs_mm")
    stats_out["gdd_decomp"] = stats.variance_decomposition(weather_year, "gdd_gs")
    stats_out["precip_gdd_spearman"] = stats.spearman_perm(
        weather_year, "precip_gs_mm", "gdd_gs"
    )
    stats_out["precip_gdd_by_year"] = {
        str(int(year)): stats.spearman_perm(sub, "precip_gs_mm", "gdd_gs", n_perm=2_000)
        for year, sub in weather_year.groupby("year")
    }
    stats_out["weather_series_groups"] = series_counts
    stats_out["weather_cells"] = (
        weather_year[["farm_slug", "year", "power_lat0", "power_lon0"]]
        .drop_duplicates()
        .groupby(["farm_slug", "year"])
        .size()
        .unstack()
        .to_dict("index")
    )

    with open(out_dir / "comparisons.json", "w") as fh:
        json.dump(stats_out, fh, indent=2, default=str)

    # ------------------------------------------------------------- figures
    ctx = {
        "farms": farms,
        "geometry": geometry,
        "weather_year": weather_year,
        "cdl_year": cdl_year,
        "composition": composition,
        "rotation": rotation,
        "cdl_matrix": cdl_matrix,
        "cdl_years": cdl_years,
        "weather_years": weather_years,
        "grower_order": grower_order,
        "grower_colors": colors,
        "crop_codes": (crop_codes, name_to_code),
        "stats": stats_out,
    }
    wanted = {s.strip().lower() for s in args.figures.split(",") if s.strip()}
    all_wanted = "all" in wanted or not wanted

    rendered, failed = [], []
    for spec in FIGURES:
        if not all_wanted and spec["id"].lower() not in wanted and spec["key"].lower() not in wanted:
            continue
        if args.skip_maps and spec["id"].endswith("-map"):
            continue
        func = getattr(figures, f"fig_{spec['key']}")
        target = fig_dir / spec["file"]
        try:
            func(ctx, target)
            rendered.append({**spec, "path": str(target.relative_to(out_dir))})
            print(f"  ok  {spec['id']:<5} {spec['file']}")
        except Exception as exc:  # keep going; a broken map must not kill tables
            failed.append({**spec, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  FAIL {spec['id']:<5} {spec['file']} -> {exc}", file=sys.stderr)

    manifest = {
        "generated_by": "my-farm-advisor/eda/eda-field-comparison/src/run_field_eda.py",
        "output_dir": str(out_dir),
        "inputs": {
            "data_pipeline_root": str(pipeline_root),
            "cdl_years": cdl_years,
            "weather_years": weather_years,
            "expected_fields_per_farm": args.expected_fields or None,
        },
        "farms": [
            {
                "grower_slug": f.grower_slug,
                "farm_slug": f.farm_slug,
                "state": f.state_abbr,
                "state_fips": f.state_fips,
                "label": f.label,
                "fields": int(len(f.fields)),
                "acres": round(f.total_acres, 1),
                "cdl_raster": str(f.cdl_raster) if f.cdl_raster else None,
                "weather_series": series_counts.get(f.key),
            }
            for f in farms
        ],
        "input_gate": gate,
        "field_labels": geometry[["grower_slug", "farm_slug", "field_id", "field_label", "area_acres"]]
        .to_dict("records"),
        "tables": {name: f"tables/{name}" for name in tables},
        "levels": {
            "vocabulary": LEVELS,
            "coverage": level_coverage(),
            "missing": [lvl for lvl, ids in level_coverage().items() if not ids],
        },
        "figures": rendered,
        "failed_figures": failed,
        "field_year_counts": {
            "weather_rows": int(len(weather_year)),
            "cdl_rows": int(len(cdl_year)),
            "fields_with_cdl": int(cdl_year["field_id"].nunique()),
            "fields_total": int(len(geometry)),
        },
    }
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    print(f"\nwrote {len(rendered)} figure(s), {len(failed)} failure(s)")
    print(f"output: {out_dir}")
    if failed:
        print("failed figures: " + ", ".join(f["id"] for f in failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

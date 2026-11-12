#!/usr/bin/env python3
"""Render a self-contained interactive Leaflet dashboard for a canonical farm.

Reads the runtime farm tree written by the data pipeline (boundaries, CDL
history, SSURGO summary, DEM summary, NDVI card summaries) and writes one HTML
file with every layer and attribute embedded. Only basemap tiles are fetched at
view time; Leaflet itself is inlined so the file opens offline.

Usage:
    python render_farm_map.py --grower il-dekalb-grower --farm dekalb-demo-farm

    python render_farm_map.py --farm-dir /path/to/farm \
        --output /tmp/farm_map.html
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import pathlib
import sys
import urllib.request

LEAFLET_VERSION = "1.9.4"
LEAFLET_JS_URL = f"https://unpkg.com/leaflet@{LEAFLET_VERSION}/dist/leaflet.js"
LEAFLET_CSS_URL = f"https://unpkg.com/leaflet@{LEAFLET_VERSION}/dist/leaflet.css"

# Categorical palette aligned with GUIDE.md crop colors, plus rotation/soil hues.
CROP_COLORS = {
    "Corn": "#2E7D32",
    "Soybeans": "#F9A825",
    "Winter Wheat": "#E65100",
    "Wheat": "#EF6C00",
    "Alfalfa": "#7B1FA2",
    "Hay": "#8E24AA",
    "Pasture": "#8D6E63",
    "Oats": "#00ACC1",
    "Barley": "#C0CA33",
    "Sorghum": "#6D4C41",
    "Cotton": "#1565C0",
    "Rice": "#0097A7",
    "Fallow": "#BDBDBD",
    "Developed": "#616161",
    "Water": "#1E88E5",
    "Forest": "#1B5E20",
    "Wetlands": "#00695C",
    "Unknown": "#9E9E9E",
    "farmland": "#607D8B",
}
FALLBACK_COLORS = [
    "#00796B", "#C2185B", "#5E35B1", "#0277BD", "#EF6C00",
    "#558B2F", "#6D4C41", "#00838F", "#AD1457", "#4527A0",
]


def lighten(hex_color: str, amount: float = 0.42) -> str:
    """Mix a hex colour toward white so fills stay legible on satellite imagery."""
    raw = hex_color.lstrip("#")
    channels = [int(raw[i: i + 2], 16) for i in (0, 2, 4)]
    mixed = [round(c + (255 - c) * amount) for c in channels]
    return "#" + "".join(f"{c:02x}" for c in mixed)


CROP_FILLS = {name: lighten(color) for name, color in CROP_COLORS.items()}

# Sequential color ramps for continuous attributes.
RAMP_GREEN = ["#F1F8E9", "#DCEDC8", "#C5E1A5", "#AED581", "#9CCC65",
              "#7CB342", "#558B2F", "#33691E", "#1B5E20"]
RAMP_EARTH = ["#FFF8E1", "#FFE0B2", "#FFCC80", "#FFB74D", "#FFA726",
              "#FB8C00", "#F57C00", "#E65100", "#BF360C"]
RAMP_TEAL = ["#E0F2F1", "#B2DFDB", "#80CBC4", "#4DB6AC", "#26A69A",
             "#00897B", "#00796B", "#00695C", "#004D40"]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def read_csv_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def discover_table_prefix(tables_dir: pathlib.Path) -> str:
    matches = sorted(tables_dir.glob("*_crop_rotation.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No *_crop_rotation.csv in {tables_dir}; run the farm pipeline first"
        )
    return matches[0].name[: -len("_crop_rotation.csv")]


def load_crop_history(tables_dir: pathlib.Path, prefix: str) -> dict[str, dict[int, str]]:
    """Dominant crop per field per year from the per-year CDL tables."""
    history: dict[str, dict[int, str]] = {}
    best_pct: dict[tuple[str, int], float] = {}
    for csv_path in sorted(tables_dir.glob(f"{prefix}_[0-9][0-9][0-9][0-9]_cdl.csv")):
        year = int(csv_path.name[len(prefix) + 1: len(prefix) + 5])
        for row in read_csv_rows(csv_path):
            field_id = row.get("field_id", "").strip()
            if not field_id:
                continue
            pct = to_float(row.get("pct")) or 0.0
            key = (field_id, year)
            if key not in best_pct or pct > best_pct[key]:
                best_pct[key] = pct
                history.setdefault(field_id, {})[year] = (
                    row.get("crop_name") or "Unknown"
                ).strip()
    return history


def load_rotation(tables_dir: pathlib.Path, prefix: str) -> dict[str, dict[str, str]]:
    path = tables_dir / f"{prefix}_crop_rotation.csv"
    return {
        row["field_id"].strip(): row
        for row in read_csv_rows(path)
        if row.get("field_id")
    }


def load_soil_summary(tables_dir: pathlib.Path, prefix: str) -> dict[str, dict[str, str]]:
    path = tables_dir / f"{prefix}_ssurgo_summary.csv"
    if not path.exists():
        return {}
    return {
        row["field_id"].strip(): row
        for row in read_csv_rows(path)
        if row.get("field_id")
    }


def load_dem_summary(tables_dir: pathlib.Path, prefix: str) -> dict[str, dict[str, str]]:
    path = tables_dir / f"{prefix}_dem_summary.csv"
    if not path.exists():
        return {}
    return {
        row["field_id"].strip(): row
        for row in read_csv_rows(path)
        if row.get("field_id")
    }


def load_ndvi(fields_dir: pathlib.Path) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for path in sorted(fields_dir.glob("*/derived/summaries/ndvi_card_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        field_id = str(payload.get("field_id") or path.parents[2].name)
        cards = payload.get("cards") or {}
        out[field_id] = {
            "corn_ndvi": to_float(str(cards.get("corn", {}).get("mean_ndvi"))),
            "soy_ndvi": to_float(str(cards.get("soybean", {}).get("mean_ndvi"))),
            "corn_peak": to_float(str(cards.get("corn_peak_95", {}).get("mean_ndvi"))),
            "soy_peak": to_float(str(cards.get("soybean_peak_95", {}).get("mean_ndvi"))),
            "corn_years": cards.get("corn", {}).get("years") or [],
            "soy_years": cards.get("soybean", {}).get("years") or [],
        }
    return out


def load_weather_span(tables_dir: pathlib.Path) -> str | None:
    matches = sorted(tables_dir.glob("*_weather_*_*.csv"))
    if not matches:
        return None
    name = matches[0].stem
    return name.split("_weather_")[-1].replace("_", "\u2013")


def newest_input_stamp(paths: list[pathlib.Path]) -> str:
    """Timestamp of the freshest pipeline input, for a deterministic footer.

    Using the newest input mtime instead of the wall clock keeps the rendered
    map byte-identical across reruns, so the farm report that links to it is
    not invalidated every time the pipeline runs.
    """
    newest = 0.0
    for path in paths:
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
    if newest <= 0:
        return "unknown"
    return _dt.datetime.fromtimestamp(newest, _dt.UTC).strftime("%Y-%m-%d %H:%M UTC")


def build_payload(farm_dir: pathlib.Path, grower: str, farm: str) -> dict:
    tables_dir = farm_dir / "derived" / "tables"
    fields_dir = farm_dir / "fields"
    boundary_path = farm_dir / "boundary" / "field_boundaries.geojson"
    if not boundary_path.exists():
        raise FileNotFoundError(f"Missing farm boundary GeoJSON: {boundary_path}")

    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    prefix = discover_table_prefix(tables_dir)

    crops = load_crop_history(tables_dir, prefix)
    rotation = load_rotation(tables_dir, prefix)
    soil = load_soil_summary(tables_dir, prefix)
    dem = load_dem_summary(tables_dir, prefix)
    ndvi = load_ndvi(fields_dir)
    weather_span = load_weather_span(tables_dir)

    crop_years = sorted({year for per_field in crops.values() for year in per_field})
    latest_year = crop_years[-1] if crop_years else None

    features = []
    for feature in boundary.get("features", []):
        props = dict(feature.get("properties") or {})
        field_id = str(props.get("field_id") or "").strip()
        if not field_id:
            continue
        history = crops.get(field_id, {})
        latest_crop = history.get(latest_year, "Unknown") if latest_year else "Unknown"
        rot = rotation.get(field_id, {})
        soil_row = soil.get(field_id, {})
        dem_row = dem.get(field_id, {})
        ndvi_row = ndvi.get(field_id, {})

        acres = to_float(str(props.get("area_acres")))
        props.update(
            {
                "field_id": field_id,
                "short_id": field_id.replace("osm-", ""),
                "county": props.get("county_name") or "",
                "state_fips": str(props.get("state_fips") or "").zfill(2),
                "area_acres": round(acres, 1) if acres is not None else None,
                "crop_history": {str(y): c for y, c in sorted(history.items())},
                "latest_crop": latest_crop,
                "rotation_sequence": rot.get("rotation_sequence", "").replace(" -> ", " \u2192 "),
                "next_crop": rot.get("predicted_next_crop", ""),
                "rotation_confidence": rot.get("rotation_confidence", ""),
                "soil_series": soil_row.get("dominant_soil", ""),
                "soil_drainage": soil_row.get("drainage_class", ""),
                "soil_om_pct": to_float(soil_row.get("avg_om_pct")),
                "soil_ph": to_float(soil_row.get("avg_ph")),
                "soil_awc_in": to_float(soil_row.get("total_aws_inches")),
                "soil_cec": to_float(soil_row.get("avg_cec")),
                "soil_clay_pct": to_float(soil_row.get("avg_clay_pct")),
                "soil_sand_pct": to_float(soil_row.get("avg_sand_pct")),
                "erosion_risk": soil_row.get("erosion_risk", ""),
                "elev_min_m": to_float(dem_row.get("elevation_min_m")),
                "elev_max_m": to_float(dem_row.get("elevation_max_m")),
                "slope_p50_pct": to_float(dem_row.get("slope_percent_p50")),
                "corn_ndvi": ndvi_row.get("corn_ndvi"),
                "soy_ndvi": ndvi_row.get("soy_ndvi"),
                "corn_peak_ndvi": ndvi_row.get("corn_peak"),
                "soy_peak_ndvi": ndvi_row.get("soy_peak"),
                "corn_years": ndvi_row.get("corn_years") or [],
                "soy_years": ndvi_row.get("soy_years") or [],
            }
        )
        features.append(
            {"type": "Feature", "geometry": feature.get("geometry"), "properties": props}
        )

    total_acres = sum((f["properties"]["area_acres"] or 0.0) for f in features)
    scenes = 0
    for manifest in fields_dir.glob("*/satellite/*/manifest.json"):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            scenes += sum(int(y.get("scene_count") or 0) for y in payload.get("years", []))
        except (OSError, ValueError):
            continue

    meta = {
        "grower": grower,
        "farm": farm,
        "farm_slug": farm_dir.name,
        "generated_at": newest_input_stamp(
            [
                boundary_path,
                *sorted(tables_dir.glob("*.csv")),
                *sorted(fields_dir.glob("*/derived/summaries/ndvi_card_summary.json")),
                *sorted(fields_dir.glob("*/satellite/*/manifest.json")),
            ]
        ),
        "field_count": len(features),
        "total_acres": round(total_acres, 1),
        "crop_years": crop_years,
        "latest_year": latest_year,
        "weather_span": weather_span,
        "satellite_scenes": scenes,
        "crop_colors": CROP_COLORS,
        "crop_fills": CROP_FILLS,
        "palette": FALLBACK_COLORS,
        "ramps": {"green": RAMP_GREEN, "earth": RAMP_EARTH, "teal": RAMP_TEAL},
    }
    return {
        "type": "FeatureCollection",
        "features": features,
        "meta": meta,
    }


# --------------------------------------------------------------------------- #
# Leaflet assets
# --------------------------------------------------------------------------- #
def fetch_text(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read().decode("utf-8")
    except Exception:  # noqa: BLE001 - offline builds fall back to CDN tags
        return None


def leaflet_assets(assets_dir: pathlib.Path) -> tuple[str | None, str | None]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    js_path = assets_dir / f"leaflet-{LEAFLET_VERSION}.js"
    css_path = assets_dir / f"leaflet-{LEAFLET_VERSION}.css"

    if not js_path.exists():
        js = fetch_text(LEAFLET_JS_URL)
        if js:
            js_path.write_text(js, encoding="utf-8")
    if not css_path.exists():
        css = fetch_text(LEAFLET_CSS_URL)
        if css:
            css_path.write_text(css, encoding="utf-8")

    js = js_path.read_text(encoding="utf-8") if js_path.exists() else None
    css = css_path.read_text(encoding="utf-8") if css_path.exists() else None
    return js, css


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>__TITLE__</title>
__LEAFLET_HEAD__
<style>
  :root {
    --ink: #14201b;
    --muted: #5b6b63;
    --line: #dfe7e2;
    --panel: #ffffff;
    --accent: #1b5e20;
    --accent-soft: #e8f1e9;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: var(--ink);
    background: #f3f5f4;
  }
  #app { display: flex; height: 100vh; overflow: hidden; }

  /* ---------- sidebar ---------- */
  #sidebar {
    width: 360px;
    flex: 0 0 360px;
    background: var(--panel);
    border-right: 1px solid var(--line);
    display: flex;
    flex-direction: column;
    z-index: 1200;
    transition: margin-left .2s ease;
  }
  #sidebar.collapsed { margin-left: -360px; }
  .side-head { padding: 20px 20px 14px; border-bottom: 1px solid var(--line); }
  .kicker { font-size: 11px; letter-spacing: .14em; text-transform: uppercase; color: var(--accent); font-weight: 700; }
  h1 { font-size: 20px; line-height: 1.25; margin: 6px 0 2px; }
  .sub { font-size: 12.5px; color: var(--muted); }
  .kpis { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 14px; }
  .kpi { background: var(--accent-soft); border-radius: 8px; padding: 8px 10px; }
  .kpi b { display: block; font-size: 16px; line-height: 1.1; }
  .kpi span { font-size: 10.5px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }

  .side-body { padding: 16px 20px; overflow-y: auto; flex: 1; }
  .control { margin-bottom: 16px; }
  .control label { display: block; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; color: var(--muted); margin-bottom: 6px; }
  select, input[type="search"] {
    width: 100%; padding: 9px 10px; border: 1px solid var(--line); border-radius: 8px;
    font-size: 13px; background: #fff; color: var(--ink);
  }
  select:focus, input[type="search"]:focus { outline: 2px solid var(--accent); outline-offset: -1px; }

  #legend { border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; }
  .legend-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; font-size: 12.5px; }
  .swatch { width: 14px; height: 14px; border-radius: 3px; flex: 0 0 14px; border: 1px solid rgba(0,0,0,.12); }
  .ramp { height: 12px; border-radius: 4px; border: 1px solid rgba(0,0,0,.12); }
  .ramp-labels { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); margin-top: 3px; }

  .section-title { display: flex; justify-content: space-between; align-items: baseline; margin: 18px 0 8px; }
  .section-title h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .07em; color: var(--muted); margin: 0; }
  .section-title span { font-size: 11px; color: var(--muted); }

  #field-list { list-style: none; margin: 0; padding: 0; }
  #field-list li {
    display: grid; grid-template-columns: 10px 1fr auto; gap: 10px; align-items: center;
    padding: 9px 8px; border-radius: 8px; cursor: pointer; border: 1px solid transparent;
  }
  #field-list li:hover { background: #f6f8f7; border-color: var(--line); }
  #field-list li.active { background: var(--accent-soft); border-color: #cfe0d1; }
  .dot { width: 10px; height: 10px; border-radius: 50%; }
  .field-name { font-size: 13px; font-weight: 600; }
  .field-sub { font-size: 11px; color: var(--muted); margin-top: 1px; }
  .acres { font-size: 12px; font-variant-numeric: tabular-nums; color: var(--muted); }

  .side-foot { padding: 12px 20px; border-top: 1px solid var(--line); font-size: 10.5px; color: var(--muted); line-height: 1.45; }

  /* ---------- map ---------- */
  #map-wrap { position: relative; flex: 1; }
  #map { position: absolute; inset: 0; background: #dfe7e2; }
  .map-ui { position: absolute; z-index: 900; }
  #basemaps { top: 14px; right: 14px; display: flex; gap: 4px; background: rgba(255,255,255,.95); padding: 4px; border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,.16); }
  #basemaps button {
    border: 0; background: transparent; padding: 6px 10px; border-radius: 7px; font-size: 11.5px;
    font-weight: 600; color: #33443c; cursor: pointer;
  }
  #basemaps button.active { background: var(--accent); color: #fff; }
  #map-tools { top: 60px; right: 14px; display: flex; flex-direction: column; gap: 6px; }
  #map-tools button, #toggle-sidebar {
    width: 34px; height: 34px; border-radius: 9px; border: 1px solid var(--line);
    background: rgba(255,255,255,.96); cursor: pointer; font-size: 15px; line-height: 1;
    box-shadow: 0 2px 10px rgba(0,0,0,.14); color: #2c3b33;
  }
  #toggle-sidebar { position: absolute; top: 14px; left: 14px; z-index: 1000; display: none; }

  .leaflet-container { font-family: inherit; }
  .leaflet-popup-content { margin: 14px 16px; width: 300px !important; }
  .pop-head { font-size: 15px; font-weight: 700; margin-bottom: 1px; }
  .pop-sub { font-size: 11.5px; color: var(--muted); margin-bottom: 10px; }
  .pop-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .pop-table td { padding: 3px 0; vertical-align: top; }
  .pop-table td:first-child { color: var(--muted); width: 46%; }
  .pop-table td:last-child { text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; }
  .pop-section { font-size: 10px; letter-spacing: .1em; text-transform: uppercase; color: var(--accent); font-weight: 700; margin: 11px 0 4px; }
  .chip { display: inline-block; padding: 2px 7px; border-radius: 20px; font-size: 10.5px; font-weight: 700; color: #fff; }
  .rotate { font-size: 12.5px; line-height: 1.45; margin-top: 2px; }
  .field-label { display: inline-block; white-space: nowrap; transform: translate(-50%, -175%); background: rgba(255,255,255,.94); border: 1px solid rgba(20,32,27,.22); border-radius: 5px; padding: 2px 6px; font-size: 10.5px; font-weight: 700; color: #23332b; box-shadow: 0 1px 3px rgba(0,0,0,.18); }
  .field-marker { background: none; border: 0; }
  .leaflet-interactive { cursor: pointer; }

  @media (max-width: 860px) {
    #sidebar { position: absolute; inset: 0 auto 0 0; box-shadow: 0 0 40px rgba(0,0,0,.25); }
    #toggle-sidebar { display: block; }
    #basemaps { top: auto; bottom: 16px; right: 14px; flex-wrap: wrap; max-width: 60vw; }
  }
</style>
</head>
<body>
<div id="app">
  <aside id="sidebar">
    <div class="side-head">
      <div class="kicker">Farm intelligence</div>
      <h1>__FARM_NAME__</h1>
      <div class="sub">__COUNTY_STATE__ &middot; __FIELD_COUNT__ fields</div>
      <div class="kpis">
        <div class="kpi"><b>__TOTAL_ACRES__</b><span>acres</span></div>
        <div class="kpi"><b>__LATEST_YEAR__</b><span>crop year</span></div>
        <div class="kpi"><b>__SCENES__</b><span>scenes</span></div>
      </div>
    </div>
    <div class="side-body">
      <div class="control">
        <label for="color-by">Color by</label>
        <select id="color-by">
          <option value="crop">Latest crop (__LATEST_YEAR__)</option>
          <option value="next_crop">Predicted next crop</option>
          <option value="soil">Dominant soil series</option>
          <option value="slope">Slope, median %</option>
          <option value="ph">Soil pH</option>
          <option value="om">Organic matter %</option>
          <option value="corn_ndvi">Corn mean NDVI</option>
          <option value="soy_ndvi">Soybean mean NDVI</option>
        </select>
      </div>
      <div class="control">
        <label>Legend</label>
        <div id="legend"></div>
      </div>
      <div class="control">
        <label for="field-search">Find field</label>
        <input id="field-search" type="search" placeholder="Filter by ID, soil, crop&hellip;" />
      </div>
      <div class="section-title"><h2>Fields</h2><span>click to zoom</span></div>
      <ul id="field-list"></ul>
    </div>
    <div class="side-foot">
      Generated __GENERATED_AT__ from the data pipeline runtime.<br />
      __SOURCE_NOTE__
    </div>
  </aside>

  <div id="map-wrap">
    <button id="toggle-sidebar" title="Toggle panel">&#9776;</button>
    <div id="map"></div>
    <div class="map-ui" id="basemaps">
      <button data-base="sat" class="active">Satellite</button>
      <button data-base="street">Street</button>
      <button data-base="topo">Terrain</button>
      <button data-base="dark">Dark</button>
    </div>
    <div class="map-ui" id="map-tools">
      <button id="zoom-fit" title="Zoom to farm">&#9974;</button>
      <button id="download-geojson" title="Download field GeoJSON">&#8681;</button>
    </div>
  </div>
</div>

<script>
const FARM = __PAYLOAD__;

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */
function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, c => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}
function fmt(value, digits = 1, suffix = "") {
  if (value === null || value === undefined || value === "" || Number.isNaN(Number(value))) return "\u2014";
  return Number(value).toFixed(digits) + suffix;
}
function hexToRgb(hex) {
  const h = hex.replace("#", "");
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}
function rampColor(hexes, t) {
  t = Math.max(0, Math.min(1, t));
  const pos = t * (hexes.length - 1);
  const i = Math.floor(pos);
  const j = Math.min(hexes.length - 1, i + 1);
  const f = pos - i;
  const a = hexToRgb(hexes[i]);
  const b = hexToRgb(hexes[j]);
  const mix = a.map((v, k) => Math.round(v + (b[k] - v) * f));
  return "#" + mix.map(v => v.toString(16).padStart(2, "0")).join("");
}
function paletteIndex(name, keys) {
  const i = keys.indexOf(name);
  return i < 0 ? 0 : i;
}

/* ------------------------------------------------------------------ */
/* color modes                                                         */
/* ------------------------------------------------------------------ */
const MODES = {
  crop:       { label: "Latest crop", kind: "cat", get: p => p.latest_crop || "Unknown" },
  next_crop:  { label: "Predicted next crop", kind: "cat", get: p => p.next_crop || "Unknown" },
  soil:       { label: "Dominant soil series", kind: "cat", get: p => p.soil_series || "Unknown" },
  slope:      { label: "Median slope %", kind: "ramp", ramp: "earth", get: p => p.slope_p50_pct, digits: 1, suffix: "%" },
  ph:         { label: "Soil pH", kind: "ramp", ramp: "teal", get: p => p.soil_ph, digits: 2 },
  om:         { label: "Organic matter %", kind: "ramp", ramp: "green", get: p => p.soil_om_pct, digits: 2, suffix: "%" },
  corn_ndvi:  { label: "Corn mean NDVI", kind: "ramp", ramp: "green", get: p => p.corn_ndvi, digits: 2 },
  soy_ndvi:   { label: "Soybean mean NDVI", kind: "ramp", ramp: "green", get: p => p.soy_ndvi, digits: 2 }
};

let mode = "crop";

function categoricalKeys() {
  const def = MODES[mode];
  const values = [...new Set(FARM.features.map(f => def.get(f.properties)))];
  values.sort((a, b) => {
    const ca = FARM.meta.crop_colors[a];
    if (ca && !FARM.meta.crop_colors[b]) return -1;
    if (!ca && FARM.meta.crop_colors[b]) return 1;
    return String(a).localeCompare(String(b));
  });
  return values;
}
function catColor(name, keys) {
  const cropFill = FARM.meta.crop_fills[name];
  if (cropFill) return cropFill;
  return FARM.meta.palette[paletteIndex(name, keys) % FARM.meta.palette.length];
}
function rampExtent() {
  const values = FARM.features
    .map(f => MODES[mode].get(f.properties))
    .filter(v => v !== null && v !== undefined && !Number.isNaN(Number(v)))
    .map(Number);
  if (!values.length) return [0, 1];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  return lo === hi ? [lo - 0.01, hi + 0.01] : [lo, hi];
}
function colorFor(props, keys) {
  const def = MODES[mode];
  if (def.kind === "cat") return catColor(def.get(props) || "Unknown", keys);
  const value = def.get(props);
  if (value === null || value === undefined) return "#B0BEC5";
  const [lo, hi] = rampExtent();
  const ramp = FARM.meta.ramps[def.ramp];
  return rampColor(ramp, (Number(value) - lo) / (hi - lo));
}

/* ------------------------------------------------------------------ */
/* map                                                                 */
/* ------------------------------------------------------------------ */
const map = L.map("map", { zoomControl: false, preferCanvas: true });
L.control.zoom({ position: "bottomright" }).addTo(map);
L.control.scale({ position: "bottomleft", imperial: true }).addTo(map);

const BASEMAPS = {
  sat: L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    attribution: "Imagery &copy; Esri, Maxar, Earthstar Geographics", maxZoom: 19
  }),
  street: L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors", maxZoom: 19
  }),
  topo: L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}", {
    attribution: "Tiles &copy; Esri", maxZoom: 19
  }),
  dark: L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution: "&copy; OpenStreetMap contributors &copy; CARTO", maxZoom: 19
  })
};
let activeBase = BASEMAPS.sat.addTo(map);

document.querySelectorAll("#basemaps button").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#basemaps button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    map.removeLayer(activeBase);
    activeBase = BASEMAPS[btn.dataset.base].addTo(map);
    activeBase.bringToBack();
  });
});

let catKeys = categoricalKeys();
let fieldsLayer = null;
let casingLayer = null;
let labelsLayer = null;
let activeFieldId = null;

function styleFor(feature) {
  const color = colorFor(feature.properties, catKeys);
  const highlighted = !activeFieldId || feature.properties.field_id === activeFieldId;
  return {
    color: highlighted ? "#12211a" : "#7c8b84",
    weight: highlighted ? 2.4 : 1.4,
    opacity: highlighted ? 1 : 0.55,
    fillColor: color,
    fillOpacity: highlighted ? 0.8 : 0.2
  };
}

function popupHtml(props) {
  const cropChip = c => `<span class="chip" style="background:${FARM.meta.crop_colors[c] || "#607D8B"}">${esc(c)}</span>`;
  const history = Object.entries(props.crop_history || {})
    .map(([year, crop]) => `<tr><td>${esc(year)}</td><td>${cropChip(crop)}</td></tr>`)
    .join("");
  const ndviRows = [
    ["Corn mean", fmt(props.corn_ndvi, 2)],
    ["Soybean mean", fmt(props.soy_ndvi, 2)],
    ["Corn peak (p95)", fmt(props.corn_peak_ndvi, 2)],
    ["Soybean peak (p95)", fmt(props.soy_peak_ndvi, 2)]
  ].map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  const soilRows = [
    ["Series", props.soil_series || "\u2014"],
    ["Drainage", props.soil_drainage || "\u2014"],
    ["Organic matter", fmt(props.soil_om_pct, 2, "%")],
    ["pH", fmt(props.soil_ph, 2)],
    ["Available water", fmt(props.soil_awc_in, 2, " in")],
    ["Clay", fmt(props.soil_clay_pct, 1, "%")],
    ["CEC", fmt(props.soil_cec, 1, " meq")],
    ["Erosion risk", props.erosion_risk || "\u2014"]
  ].map(([k, v]) => `<tr><td>${k}</td><td>${esc(v)}</td></tr>`).join("");
  const terrainRows = [
    ["Elevation", `${fmt(props.elev_min_m, 0)} \u2013 ${fmt(props.elev_max_m, 0)} m`],
    ["Median slope", fmt(props.slope_p50_pct, 2, "%")]
  ].map(([k, v]) => `<tr><td>${k}</td><td>${esc(v)}</td></tr>`).join("");

  return `
    <div class="pop-head">Field ${esc(props.short_id || props.field_id)}</div>
    <div class="pop-sub">${esc(props.county)} County &middot; ${fmt(props.area_acres, 1)} acres</div>
    <div class="pop-section">Crop history</div>
    <table class="pop-table">${history}</table>
    <div class="rotate"><b>Rotation:</b> ${esc(props.rotation_sequence || "\u2014")}<br />
      <b>Next:</b> ${cropChip(props.next_crop || "Unknown")}
      <span style="color:#5b6b63"> (${esc(props.rotation_confidence || "n/a")} confidence)</span>
    </div>
    <div class="pop-section">Soil</div>
    <table class="pop-table">${soilRows}</table>
    <div class="pop-section">Terrain</div>
    <table class="pop-table">${terrainRows}</table>
    <div class="pop-section">NDVI (2021\u20132025)</div>
    <table class="pop-table">${ndviRows}</table>
  `;
}

function buildLabels() {
  if (labelsLayer) map.removeLayer(labelsLayer);
  labelsLayer = L.layerGroup();
  FARM.features.forEach(feature => {
    const p = feature.properties;
    const center = L.geoJSON(feature).getBounds().getCenter();
    // A dot keeps small fields identifiable at farm-extent zoom, where the
    // polygon itself can be only a few pixels wide.
    labelsLayer.addLayer(L.circleMarker(center, {
      radius: 6,
      color: "#ffffff",
      weight: 2.5,
      opacity: 1,
      fillColor: colorFor(p, catKeys),
      fillOpacity: 1,
      interactive: false
    }));
    labelsLayer.addLayer(L.marker(center, {
      interactive: false,
      icon: L.divIcon({
        className: "field-marker",
        html: `<div class="field-label">${esc(p.short_id)} &middot; ${fmt(p.area_acres, 0)} ac &middot; ${esc(p.latest_crop)}</div>`,
        iconSize: [0, 0],
        iconAnchor: [0, 0]
      })
    }));
  });
  labelsLayer.addTo(map);
}

function buildLayers() {
  if (fieldsLayer) map.removeLayer(fieldsLayer);
  if (labelsLayer) map.removeLayer(labelsLayer);
  if (casingLayer) map.removeLayer(casingLayer);

  // White casing under the crop/soil fill keeps boundaries legible on imagery.
  casingLayer = L.geoJSON(FARM, {
    style: { color: "#ffffff", weight: 5, opacity: 0.8, fillOpacity: 0 },
    interactive: false
  }).addTo(map);

  fieldsLayer = L.geoJSON(FARM, {
    style: styleFor,
    onEachFeature: (feature, layer) => {
      layer.bindPopup(popupHtml(feature.properties), {
        maxWidth: 340,
        autoPanPadding: [30, 30],
        keepInView: true
      });
      layer.on("click", () => { setActiveField(feature.properties.field_id); });
      layer.on("mouseover", () => layer.setStyle({ weight: 3.4 }));
      layer.on("mouseout", () => layer.setStyle(styleFor(feature)));
    }
  }).addTo(map);

  buildLabels();

  if (!activeFieldId) {
    map.fitBounds(fieldsLayer.getBounds().pad(0.12));
  }
}

/* ------------------------------------------------------------------ */
/* legend + field list                                                 */
/* ------------------------------------------------------------------ */
function updateLegend() {
  const def = MODES[mode];
  const box = document.getElementById("legend");
  if (def.kind === "cat") {
    catKeys = categoricalKeys();
    box.innerHTML = catKeys.map(name => `
      <div class="legend-row">
        <span class="swatch" style="background:${catColor(name, catKeys)}"></span>
        <span>${esc(name)}</span>
      </div>`).join("");
  } else {
    const ramp = FARM.meta.ramps[def.ramp];
    const [lo, hi] = rampExtent();
    box.innerHTML = `
      <div class="ramp" style="background:linear-gradient(90deg, ${ramp.join(", ")})"></div>
      <div class="ramp-labels"><span>${fmt(lo, def.digits ?? 1, def.suffix ?? "")}</span>
      <span>${fmt(hi, def.digits ?? 1, def.suffix ?? "")}</span></div>
      <div class="ramp-labels" style="margin-top:6px"><span>${esc(def.label)}</span><span>per field</span></div>`;
  }
}

function renderFieldList() {
  const query = (document.getElementById("field-search").value || "").toLowerCase();
  const list = document.getElementById("field-list");
  const rows = FARM.features
    .map(f => f.properties)
    .filter(p => !query || [p.field_id, p.latest_crop, p.soil_series, p.next_crop]
      .join(" ").toLowerCase().includes(query))
    .sort((a, b) => (b.area_acres || 0) - (a.area_acres || 0));

  list.innerHTML = rows.map(p => `
    <li data-id="${esc(p.field_id)}" class="${p.field_id === activeFieldId ? "active" : ""}">
      <span class="dot" style="background:${colorFor(p, catKeys)}"></span>
      <span>
        <div class="field-name">${esc(p.short_id)}</div>
        <div class="field-sub">${esc(p.latest_crop)} &middot; ${esc(p.soil_series || "soil n/a")}</div>
      </span>
      <span class="acres">${fmt(p.area_acres, 0)} ac</span>
    </li>`).join("") || `<li style="cursor:default;color:#5b6b63">No matching fields</li>`;

  list.querySelectorAll("li[data-id]").forEach(li => {
    li.addEventListener("click", () => {
      const id = li.dataset.id;
      setActiveField(id);
      const layer = fieldsLayer.getLayers().find(l => l.feature.properties.field_id === id);
      if (layer) {
        map.fitBounds(layer.getBounds().pad(0.6), { maxZoom: 16 });
        // Open after the fly-to settles so Leaflet can auto-pan the popup.
        setTimeout(() => layer.openPopup(), 420);
      }
    });
  });
}

function setActiveField(fieldId) {
  activeFieldId = fieldId;
  fieldsLayer.eachLayer(layer => layer.setStyle(styleFor(layer.feature)));
  renderFieldList();
}

/* ------------------------------------------------------------------ */
/* controls                                                            */
/* ------------------------------------------------------------------ */
document.getElementById("color-by").addEventListener("change", event => {
  mode = event.target.value;
  catKeys = categoricalKeys();
  fieldsLayer.eachLayer(layer => layer.setStyle(styleFor(layer.feature)));
  buildLabels();
  updateLegend();
  renderFieldList();
});

document.getElementById("field-search").addEventListener("input", renderFieldList);

document.getElementById("zoom-fit").addEventListener("click", () => {
  activeFieldId = null;
  fieldsLayer.eachLayer(layer => layer.setStyle(styleFor(layer.feature)));
  buildLabels();
  map.fitBounds(fieldsLayer.getBounds().pad(0.12));
  renderFieldList();
});

document.getElementById("download-geojson").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(FARM, null, 2)], { type: "application/geo+json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = FARM.meta.farm_slug + "_fields.geojson";
  anchor.click();
  URL.revokeObjectURL(url);
});

document.getElementById("toggle-sidebar").addEventListener("click", () => {
  document.getElementById("sidebar").classList.toggle("collapsed");
});

/* ------------------------------------------------------------------ */
/* boot                                                                */
/* ------------------------------------------------------------------ */
buildLayers();
updateLegend();
renderFieldList();
</script>
</body>
</html>
"""


def build_html(payload: dict, leaflet_js: str | None, leaflet_css: str | None) -> str:
    meta = payload["meta"]
    county = sorted({f["properties"]["county"] for f in payload["features"] if f["properties"]["county"]})
    state_names = {"17": "Illinois", "19": "Iowa"}
    states = sorted({state_names.get(f["properties"]["state_fips"], f["properties"]["state_fips"])
                     for f in payload["features"]})
    county_state = ", ".join(county) + (" County" if county else "") + " &middot; " + ", ".join(states)

    if leaflet_js and leaflet_css:
        head = f"<style>{leaflet_css}</style>\n<script>{leaflet_js}</script>"
    else:
        head = (
            f'<link rel="stylesheet" href="{LEAFLET_CSS_URL}" />\n'
            f'<script src="{LEAFLET_JS_URL}"></script>'
        )

    source_note = (
        "Boundaries: OpenStreetMap via Overpass &middot; Crops: USDA CDL "
        f"{meta['crop_years'][0] if meta['crop_years'] else ''}"
        f"\u2013{meta['latest_year'] or ''} &middot; Soil: USDA SSURGO "
        "&middot; Terrain: USGS 3DEP &middot; NDVI: Sentinel-2 / Landsat"
    )

    replacements = {
        "__TITLE__": f"{meta['farm']} \u2014 Interactive Field Map",
        "__LEAFLET_HEAD__": head,
        "__FARM_NAME__": meta["farm"].replace("-", " ").title(),
        "__COUNTY_STATE__": county_state,
        "__FIELD_COUNT__": str(meta["field_count"]),
        "__TOTAL_ACRES__": f"{meta['total_acres']:,.0f}",
        "__LATEST_YEAR__": str(meta["latest_year"] or "\u2014"),
        "__SCENES__": f"{meta['satellite_scenes']:,}",
        "__GENERATED_AT__": meta["generated_at"],
        "__SOURCE_NOTE__": source_note,
        "__PAYLOAD__": json.dumps(payload, separators=(",", ":")),
    }
    html = HTML_TEMPLATE
    for token, value in replacements.items():
        html = html.replace(token, value)
    return html


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def default_data_root() -> str:
    """Resolve the runtime data-pipeline directory.

    The pipeline contract sets ``DATA_PIPELINE_DATA_ROOT`` to the runtime root,
    with ``<root>/data-pipeline`` holding ``growers/`` and ``shared/``. Accept a
    path that already points at the data-pipeline directory too.
    """
    fallback = pathlib.Path.home() / "my-farm-advisor-runtime" / "data-pipeline"
    raw_root = os.environ.get("DATA_PIPELINE_DATA_ROOT")
    if not raw_root:
        return str(fallback)
    candidate = pathlib.Path(raw_root).expanduser()
    if (candidate / "data-pipeline").is_dir():
        candidate = candidate / "data-pipeline"
    return str(candidate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grower", default=None, help="Grower slug")
    parser.add_argument("--farm", default=None, help="Farm slug")
    parser.add_argument("--farm-dir", default=None, help="Explicit farm directory")
    parser.add_argument(
        "--data-root",
        default=default_data_root(),
        help="Runtime data-pipeline root (contains growers/ and shared/)",
    )
    parser.add_argument("--output", default=None, help="Output HTML path")
    parser.add_argument(
        "--assets-dir",
        default=None,
        help="Leaflet cache dir (defaults to <data-root>/shared/vendor)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = pathlib.Path(args.data_root).expanduser().resolve()

    if args.farm_dir:
        farm_dir = pathlib.Path(args.farm_dir).expanduser().resolve()
    elif args.grower and args.farm:
        farm_dir = data_root / "growers" / args.grower / "farms" / args.farm
    else:
        print("Provide --farm-dir or both --grower and --farm", file=sys.stderr)
        return 2

    if not farm_dir.exists():
        print(f"Farm directory not found: {farm_dir}", file=sys.stderr)
        return 2

    grower = args.grower or farm_dir.parents[1].name
    farm = args.farm or farm_dir.name

    payload = build_payload(farm_dir, grower, farm)
    assets_dir = pathlib.Path(args.assets_dir).expanduser() if args.assets_dir else data_root / "shared" / "vendor"
    leaflet_js, leaflet_css = leaflet_assets(assets_dir)

    html = build_html(payload, leaflet_js, leaflet_css)
    output = (
        pathlib.Path(args.output).expanduser()
        if args.output
        else farm_dir / "derived" / "reports" / f"{farm_dir.name}_interactive_map.html"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    print(
        json.dumps(
            {
                "farm": farm,
                "fields": payload["meta"]["field_count"],
                "acres": payload["meta"]["total_acres"],
                "leaflet_inlined": bool(leaflet_js and leaflet_css),
                "output": str(output),
                "size_kb": round(output.stat().st_size / 1024, 1),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

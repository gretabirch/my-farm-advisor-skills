"""Loaders and derived panels for field-level EDA.

Reads the runtime farm tree described by ``DATA_PIPELINE_DATA_ROOT`` and
produces tidy, analysis-ready tables:

  * field geometry metrics (equal-area projection)
  * weather field-year metrics (growing season May-Sep)
  * CDL field-year composition and coverage

No files are written here; ``run_field_eda.py`` owns output paths.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

GROWING_MONTHS = (5, 6, 7, 8, 9)
ACRE_M2 = 4046.8564224
EQUAL_AREA_CRS = "EPSG:5070"

# NASA POWER native grid: 0.5 deg latitude x 0.625 deg longitude, with cell
# CENTRES at lat = 0.25 + 0.5k and lon = 0.3125 + 0.625k. Verified against the
# runtime data: this registration reproduces the observed identical-series
# groups in all three farms exactly (see GUIDE.md "weather resolution").
POWER_DLAT = 0.5
POWER_DLON = 0.625

STATE_ABBR = {
    "17": "IL", "19": "IA", "31": "NE", "27": "MN", "18": "IN", "20": "KS",
    "29": "MO", "38": "ND", "46": "SD", "55": "WI", "26": "MI", "39": "OH",
}


# --------------------------------------------------------------------------
# path resolution
# --------------------------------------------------------------------------

def runtime_pipeline_root() -> Path:
    raw = os.environ.get("DATA_PIPELINE_DATA_ROOT", "~/my-farm-advisor-runtime")
    root = Path(raw).expanduser() / "data-pipeline"
    if not root.is_dir():
        raise SystemExit(
            f"Runtime data not found at {root}. Set DATA_PIPELINE_DATA_ROOT to the "
            "directory that contains data-pipeline/."
        )
    return root


def repo_lib_dir() -> Path:
    """Field-frame/basemap helpers: prefer the runtime pipeline copy."""
    data_root = os.environ.get("DATA_PIPELINE_DATA_ROOT")
    if data_root:
        candidate = (
            Path(data_root).expanduser() / "data-pipeline" / "src" / "scripts" / "lib"
        )
        if candidate.is_dir():
            return candidate
    return Path(__file__).resolve().parents[3] / "data-pipeline" / "src" / "scripts" / "lib"


def _ensure_lib_on_path() -> None:
    lib = repo_lib_dir()
    if str(lib) not in sys.path:
        sys.path.insert(0, str(lib))


# --------------------------------------------------------------------------
# dataset records
# --------------------------------------------------------------------------

@dataclass
class Farm:
    grower_slug: str
    farm_slug: str
    farm_dir: Path
    fields: gpd.GeoDataFrame
    weather: pd.DataFrame
    cdl: pd.DataFrame

    @property
    def key(self) -> str:
        return f"{self.grower_slug}/{self.farm_slug}"

    @property
    def state_fips(self) -> str:
        return str(self.fields["state_fips"].iloc[0]).zfill(2)

    @property
    def state_abbr(self) -> str:
        return STATE_ABBR.get(self.state_fips, self.state_fips)

    @property
    def label(self) -> str:
        return f"{self.state_abbr} · {self.farm_slug}"

    @property
    def short_label(self) -> str:
        return self.state_abbr

    @property
    def total_acres(self) -> float:
        return float(self.fields["area_acres"].sum())

    @property
    def cdl_raster(self) -> Path | None:
        """Latest CDL state raster available for the map panel."""
        root = self.farm_dir.parents[3] / "shared" / "cdl" / "rasters"
        year = int(self.cdl["year"].max()) if len(self.cdl) else 2023
        path = root / f"CDL_{year}_{int(self.state_fips)}.tif"
        return path if path.exists() else None


def discover_farms(pipeline_root: Path, growers: list[str] | None = None) -> list[Farm]:
    grower_dirs = sorted(p for p in (pipeline_root / "growers").glob("*") if p.is_dir())
    if growers:
        wanted = {g.strip() for g in growers if g.strip()}
        grower_dirs = [g for g in grower_dirs if g.name in wanted]
        missing = wanted - {g.name for g in grower_dirs}
        if missing:
            raise SystemExit(f"Unknown grower slug(s): {', '.join(sorted(missing))}")

    farms: list[Farm] = []
    for grower_dir in grower_dirs:
        for farm_dir in sorted((grower_dir / "farms").glob("*")):
            boundary = farm_dir / "boundary" / "field_boundaries.geojson"
            tables = farm_dir / "derived" / "tables"
            weather_csvs = (
                sorted(tables.glob("*_weather_2021_2025.csv")) if tables.is_dir() else []
            )
            cdl_csvs = (
                sorted(tables.glob("*_cdl_*_full_composition.csv"))
                if tables.is_dir()
                else []
            )
            if not (boundary.exists() and weather_csvs and cdl_csvs):
                continue
            farms.append(
                Farm(
                    grower_slug=grower_dir.name,
                    farm_slug=farm_dir.name,
                    farm_dir=farm_dir,
                    fields=load_fields(boundary),
                    weather=load_weather(weather_csvs[0]),
                    cdl=load_cdl(cdl_csvs[0]),
                )
            )
    if not farms:
        raise SystemExit(f"No complete farms found under {pipeline_root / 'growers'}")
    return farms


# --------------------------------------------------------------------------
# boundaries
# --------------------------------------------------------------------------

def load_fields(boundary_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(boundary_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if "field_id" not in gdf.columns or "area_acres" not in gdf.columns:
        raise SystemExit(f"{boundary_path}: expected field_id and area_acres columns")

    proj = gdf.to_crs(EQUAL_AREA_CRS)
    area_m2 = proj.geometry.area
    perimeter_m = proj.geometry.length
    hull_area = proj.geometry.convex_hull.area

    def _aspect(geom) -> float:
        if geom.is_empty:
            return np.nan
        mbr = geom.minimum_rotated_rectangle
        if mbr.is_empty or mbr.geom_type != "Polygon":
            return np.nan
        xs, ys = mbr.exterior.xy
        edges = [
            ((xs[i + 1] - xs[i]) ** 2 + (ys[i + 1] - ys[i]) ** 2) ** 0.5
            for i in range(4)
        ]
        short, long = min(edges), max(edges)
        return float(long / short) if short > 0 else np.nan

    centroid = proj.geometry.centroid
    out = gdf.copy()
    out["area_m2_calc"] = area_m2.values
    out["area_ha"] = area_m2.values / 10_000.0
    out["perimeter_m"] = perimeter_m.values
    out["compactness_pp"] = (4.0 * np.pi * area_m2 / perimeter_m.pow(2)).values
    out["convexity"] = (area_m2 / hull_area.where(hull_area > 0)).values
    out["aspect_ratio"] = proj.geometry.apply(_aspect).values
    out["centroid_x_m"] = centroid.x.values
    out["centroid_y_m"] = centroid.y.values
    out["state_fips"] = out["state_fips"].astype(str).str.zfill(2)
    return out


def fields_geometry_table(farms: list[Farm]) -> pd.DataFrame:
    """One row per field, with a stable human label ordered by area."""
    frames = []
    for farm in farms:
        df = farm.fields[
            [
                "field_id", "state_fips", "county_name", "area_acres", "area_ha",
                "perimeter_m", "compactness_pp", "convexity", "aspect_ratio",
                "centroid_x_m", "centroid_y_m",
            ]
        ].copy()
        df = df.sort_values("area_acres", ascending=False).reset_index(drop=True)
        df.insert(0, "field_label", [f"{farm.short_label}-{i + 1:02d}" for i in range(len(df))])
        df.insert(0, "farm_slug", farm.farm_slug)
        df.insert(0, "grower_slug", farm.grower_slug)
        df["grower_label"] = farm.short_label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# weather
# --------------------------------------------------------------------------

def load_weather(weather_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(weather_csv, parse_dates=["date"])
    required = {"field_id", "date", "T2M_MAX", "T2M_MIN", "PRECTOTCORR"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{weather_csv}: missing weather columns {sorted(missing)}")
    df["field_id"] = df["field_id"].astype(str)
    df = df.sort_values(["field_id", "date"]).reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["doy"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["in_season"] = df["month"].isin(GROWING_MONTHS)
    t_avg = (df["T2M_MAX"] + df["T2M_MIN"]) / 2.0
    df["gdd"] = np.maximum(0.0, t_avg - 10.0)
    df["gdd_cumulative"] = df.groupby(["field_id", "year"])["gdd"].cumsum()
    if "lat" in df.columns and "lon" in df.columns:
        cells = df[["lat", "lon"]].apply(
            lambda r: power_cell(float(r["lat"]), float(r["lon"])), axis=1
        )
        df["power_lat0"] = [c[0] for c in cells]
        df["power_lon0"] = [c[1] for c in cells]
    return df


def power_cell(lat: float, lon: float) -> tuple[float, float]:
    """SW corner of the NASA POWER cell containing (lat, lon)."""
    lat0 = np.floor((lat - 0.25) / POWER_DLAT) * POWER_DLAT + 0.25
    lon0 = np.floor((lon - 0.3125) / POWER_DLON) * POWER_DLON + 0.3125
    return float(lat0), float(lon0)


def _frost_dates(group: pd.DataFrame) -> tuple[float, float]:
    cold = group[group["T2M_MIN"] <= 0.0]
    spring = cold[cold["doy"] <= 213]["doy"]
    fall = cold[cold["doy"] >= 214]["doy"]
    return (
        float(spring.max()) if not spring.empty else np.nan,
        float(fall.min()) if not fall.empty else np.nan,
    )


def weather_field_year(farms: list[Farm], geometry: pd.DataFrame) -> pd.DataFrame:
    areas = geometry.set_index("field_id")["area_acres"]
    labels = geometry.set_index("field_id")["field_label"]
    rows = []
    for farm in farms:
        for (field_id, year), group in farm.weather.groupby(["field_id", "year"], sort=True):
            season = group[group["in_season"]]
            last_frost, first_frost = _frost_dates(group)
            rows.append(
                {
                    "grower_slug": farm.grower_slug,
                    "farm_slug": farm.farm_slug,
                    "grower_label": farm.short_label,
                    "field_id": field_id,
                    "field_label": labels.get(field_id, field_id),
                    "field_area_acres": float(areas.get(field_id, np.nan)),
                    "year": int(year),
                    "precip_gs_mm": float(season["PRECTOTCORR"].sum()),
                    "precip_year_mm": float(group["PRECTOTCORR"].sum()),
                    "gdd_gs": float(season["gdd"].sum()),
                    "gdd_year": float(group["gdd"].sum()),
                    "tmean_gs_c": float(
                        season["T2M_MAX"].mean() + season["T2M_MIN"].mean()
                    ) / 2.0,
                    "tmax_gs_c": float(season["T2M_MAX"].mean()),
                    "tmin_gs_c": float(season["T2M_MIN"].mean()),
                    "days": int(len(group)),
                    "last_spring_frost_doy": last_frost,
                    "first_fall_frost_doy": first_frost,
                    "power_lat0": float(group["power_lat0"].iloc[0])
                    if "power_lat0" in group
                    else np.nan,
                    "power_lon0": float(group["power_lon0"].iloc[0])
                    if "power_lon0" in group
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def weather_series_counts(farms: list[Farm]) -> dict[str, int]:
    """Distinct daily series per farm: reveals the POWER grid resolution."""
    counts: dict[str, int] = {}
    for farm in farms:
        wide = farm.weather.pivot_table(
            index="date", columns="field_id", values="PRECTOTCORR"
        )
        counts[farm.key] = int(wide.T.drop_duplicates().shape[0])
    return counts


# --------------------------------------------------------------------------
# CDL
# --------------------------------------------------------------------------

def load_cdl(cdl_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(cdl_csv)
    required = {"field_id", "year", "crop_code", "crop_name", "pct"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{cdl_csv}: missing CDL columns {sorted(missing)}")
    df["field_id"] = df["field_id"].astype(str)
    df["year"] = df["year"].astype(int)
    df["crop_code"] = df["crop_code"].astype(int)
    df["pct"] = pd.to_numeric(df["pct"], errors="coerce").fillna(0.0)
    return df


def cdl_field_year(farms: list[Farm], geometry: pd.DataFrame) -> pd.DataFrame:
    labels = geometry.set_index("field_id")["field_label"]
    rows = []
    for farm in farms:
        for (field_id, year), group in farm.cdl.groupby(["field_id", "year"], sort=True):
            ranked = group.sort_values(["pct", "crop_name"], ascending=[False, True])
            top = ranked.iloc[0]
            rows.append(
                {
                    "grower_slug": farm.grower_slug,
                    "farm_slug": farm.farm_slug,
                    "grower_label": farm.short_label,
                    "field_id": field_id,
                    "field_label": labels.get(field_id, field_id),
                    "year": int(year),
                    "dominant_crop": str(top["crop_name"]),
                    "dominant_code": int(top["crop_code"]),
                    "dominant_pct": float(top["pct"]),
                    "cdl_coverage_pct": float(group["pct"].sum()),
                    "classes": int((group["pct"] >= 1.0).sum()),
                }
            )
    return pd.DataFrame(rows)


def cdl_composition_mean(farms: list[Farm], geometry: pd.DataFrame) -> pd.DataFrame:
    """Mean share per field x class over the years that field has data.

    Normalised to 100 % per field so the stacked bars always fill exactly one
    unit height; the unnormalised sum (= mean CDL coverage) is reported in
    ``cdl_coverage_pct`` on the field-year table.
    """
    labels = geometry.set_index("field_id")["field_label"]
    frames = []
    for farm in farms:
        df = farm.cdl.copy()
        mean = df.groupby(["field_id", "crop_name", "crop_code"], as_index=False)["pct"].mean()
        mean["years_observed"] = mean["field_id"].map(
            df.groupby("field_id")["year"].nunique()
        )
        totals = mean.groupby("field_id")["pct"].transform("sum").replace(0, np.nan)
        mean["pct_mean_norm"] = (mean["pct"] / totals * 100.0).fillna(0.0)
        mean.insert(0, "farm_slug", farm.farm_slug)
        mean.insert(0, "grower_slug", farm.grower_slug)
        mean.insert(2, "grower_label", farm.short_label)
        mean.insert(4, "field_label", mean["field_id"].map(labels))
        frames.append(mean)
    return pd.concat(frames, ignore_index=True)


def rotation_metrics(geometry: pd.DataFrame, cdl_year: pd.DataFrame) -> pd.DataFrame:
    """Crop changes and diversity per field (fields without CDL stay absent)."""
    rows = []
    for (grower, farm, field_id), group in cdl_year.groupby(
        ["grower_slug", "farm_slug", "field_id"], sort=True
    ):
        group = group.sort_values("year")
        seq = group["dominant_crop"].tolist()
        changes = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
        counts = group["dominant_crop"].value_counts(normalize=True)
        shannon = float(-(counts * np.log(counts)).sum()) if len(counts) else np.nan
        rows.append(
            {
                "grower_slug": grower,
                "farm_slug": farm,
                "field_id": field_id,
                "years_with_cdl": int(len(group)),
                "crop_changes": int(changes),
                "crop_diversity_shannon": shannon,
                "dominant_sequence": " > ".join(seq),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics = metrics.merge(
        geometry[["grower_slug", "farm_slug", "field_id", "field_label",
                  "grower_label", "area_acres"]],
        on=["grower_slug", "farm_slug", "field_id"],
        how="left",
    )
    return metrics


def cdl_matrix(
    farms: list[Farm], geometry: pd.DataFrame, years: list[int]
) -> pd.DataFrame:
    """Fields x years dominant-crop matrix; missing CDL stays NaN."""
    labels = geometry.set_index("field_id")["field_label"]
    rows = []
    for farm in farms:
        for field_id in farm.fields["field_id"].astype(str):
            row = {
                "grower_slug": farm.grower_slug,
                "farm_slug": farm.farm_slug,
                "grower_label": farm.short_label,
                "field_id": field_id,
                "field_label": labels.get(field_id, field_id),
            }
            sub = farm.cdl[farm.cdl["field_id"] == field_id]
            for year in years:
                hit = sub[sub["year"] == year]
                row[year] = (
                    hit.sort_values("pct", ascending=False)["crop_name"].iloc[0]
                    if not hit.empty
                    else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# input gate
# --------------------------------------------------------------------------

def validate_inputs(
    farms: list[Farm],
    *,
    expected_fields: int | None,
    cdl_years: list[int],
    weather_years: list[int],
) -> dict:
    problems: list[str] = []
    notes: list[str] = []
    per_farm = []
    expected_days = len(
        pd.date_range(f"{weather_years[0]}-01-01", f"{weather_years[-1]}-12-31")
    )
    for farm in farms:
        n_fields = len(farm.fields)
        if expected_fields is not None and n_fields != expected_fields:
            problems.append(f"{farm.key}: {n_fields} fields, expected {expected_fields}")

        weather_counts = farm.weather.groupby("field_id")["date"].nunique()
        bad_weather = weather_counts[weather_counts != expected_days]
        if not bad_weather.empty:
            problems.append(
                f"{farm.key}: {len(bad_weather)} field(s) missing weather days "
                f"(expected {expected_days}, got {sorted(set(weather_counts))})"
            )

        found_years = sorted(int(y) for y in farm.cdl["year"].unique())
        missing_years = sorted(set(cdl_years) - set(found_years))
        if missing_years:
            problems.append(f"{farm.key}: CDL years missing {missing_years}")

        cdl_ids = set(farm.cdl["field_id"].astype(str))
        no_cdl = sorted(set(farm.fields["field_id"].astype(str)) - cdl_ids)
        if no_cdl:
            notes.append(
                f"{farm.key}: no CDL rows for {len(no_cdl)} field(s): "
                + ", ".join(no_cdl)
            )

        per_farm.append(
            {
                "farm": farm.key,
                "label": farm.label,
                "state_fips": farm.state_fips,
                "fields": n_fields,
                "acres": round(farm.total_acres, 1),
                "cdl_rows": int(len(farm.cdl)),
                "cdl_years": found_years,
                "weather_rows": int(len(farm.weather)),
                "weather_days_per_field": int(expected_days),
                "fields_without_cdl": no_cdl,
            }
        )

    if problems:
        raise SystemExit("Input gate failed:\n  - " + "\n  - ".join(problems))
    return {"farms": per_farm, "notes": notes, "expected_weather_days": expected_days}

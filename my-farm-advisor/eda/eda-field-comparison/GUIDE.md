---
name: eda-field-comparison
description: Field-level EDA across one or more growers - parcel geometry, CDL land cover and rotation, and growing-season weather - producing twelve static figures, CSV tables, a comparisons JSON, and a manifest. Use when comparing fields within a grower, growers against each other, or a field against itself over time.
version: 1.0.0
author: Boreal Bytes
tags: [eda, fields, boundaries, cdl, weather, rotation, comparison]
---

# Workflow: eda-field-comparison

## Description

Answers the three questions a grower asks about a field portfolio, with
boundaries, CDL, and weather only (no soil):

| Level | Question |
|---|---|
| within field, over time | does this field change, and how much? |
| across fields, within grower | which parcels carry the farm's risk and its acres? |
| across growers | what is actually different, and what is only weather? |

Each level gets two statistical visuals and one comparison/correlation, plus a
locator map: twelve figures, one story each.

## When to use this workflow

- "Compare our three demo farms" / "how do my fields differ from the
  neighbours'?"
- An acreage, shape, land-cover, rotation, precipitation, or GDD question that
  spans more than one field.
- A first causal check before yield or management analysis: is an apparent
  field difference just a shared weather grid cell?

Do **not** use it for soil, imagery indices, or yield — those belong to
`soil/`, `imagery/`, `field-management/`.

## Prerequisites

- `DATA_PIPELINE_DATA_ROOT` pointing at a pipeline root with
  `growers/<grower>/farms/<farm>/` (boundaries), `shared/cdl/` (CDL CSV +
  rasters) and `shared/weather/` (daily weather CSV).
- The pipeline virtualenv (matplotlib, geopandas, rasterio, contextily,
  pyproj). The examples below use `${DATA_PIPELINE_DATA_ROOT}/data-pipeline/.venv`.

## Quick start

```bash
cd <repo>
DATA_PIPELINE_DATA_ROOT=$HOME/my-farm-advisor-runtime \
  $HOME/my-farm-advisor-runtime/data-pipeline/.venv/bin/python \
  my-farm-advisor/eda/eda-field-comparison/src/run_field_eda.py \
  --output-dir $HOME/my-farm-advisor-runtime/data-pipeline/eda/my-field-eda
```

Add `--skip-maps` to iterate quickly on the nine statistical panels (the three
maps re-fetch satellite tiles). Add `--figures A1,C3` to rebuild a subset.
`--growers il,ia` limits the run to a subset of grower slugs.

## Outputs

```
<output-dir>/
  figures/           12 PNGs, numbered in reading order
  tables/            fields_geometry.csv, cdl_field_year.csv, cdl_matrix.csv,
                     cdl_composition_mean.csv, field_rotation.csv,
                     weather_field_year.csv
  comparisons.json   every test result the figures quote
  manifest.json      inputs, versions, figure index, story per figure
```

`manifest.json` is the hand-off contract for a report: it names each figure
with its story, caveat, and comparison level, so a document can be assembled
without re-reading the data.

## The twelve figures

| ID | Figure | Level | Story |
|---|---|---|---|
| A1 | field area by grower | across growers | field size spans two orders of magnitude |
| A2 | area vs compactness | across growers | the biggest parcels are the most irregular; the smallest are only a few pixels wide |
| A3 | CDL coverage | comparison | coverage gaps sit on the smallest parcels, not on a year trend |
| A-map | farm boundary maps | locator | fields are scattered parcels, not one block |
| B1 | CDL composition per field | within grower | mean crop mix hides the rotation |
| B2 | dominant class, field × year | within field | rotation is field-specific |
| B3 | rotation change vs diversity | comparison | rotation diversity has no grower signal and no size trend |
| B-map | CDL 2023 mosaic | locator | the mosaic continues inside every parcel |
| C1 | growing-season precipitation | across growers | year-to-year swings dwarf field differences |
| C2 | cumulative GDD | within grower | one or two grid cells per farm, never one per field |
| C3 | variance decomposition + precip/GDD scatter | comparison | rain is year-driven, heat is grower-driven |
| C-map | NASA POWER cells | locator | weather is sampled on a ~50 km grid |

Story text lives in `FIGURES` in `src/run_field_eda.py`; change it there, not in
the prose of downstream documents.

## How the analysis works

- **Geometry** (`eda_data.fields_geometry_table`): geodesic acres, perimeter,
  Polsby-Popper compactness, convexity, aspect ratio, centroid.
- **Weather** (`weather_field_year`): growing season is May 1 – Sep 30; GDD is
  base 10 °C; each field inherits the NASA POWER 0.5° × 0.625° cell containing
  its centroid, so fields in one cell share an identical daily series.
- **CDL** (`cdl_field_year`, `field_rotation`, `cdl_matrix`): per-year dominant class,
  classified-pixel coverage, transitions between consecutive years, and a
  Shannon diversity over the field's year-class mix.
- **Tests** (`eda_stats`): Kruskal-Wallis with ε², pairwise Mann-Whitney with
  Holm correction, Spearman with a permutation p-value and a bootstrap CI
  (seed 20261120), and a sequential nested variance decomposition
  (year → grower → field within grower → residual).

With three growers and ten fields each, every p-value is descriptive. The
figures say so; keep it that way when reusing the numbers.

## Field-frame contract for maps

All three maps reuse `data-pipeline/src/scripts/lib/field_frame.py` and
`basemap.py`: local-metre UTM axes, auto-rotation, boxed north arrow, and the
Esri underlay. Two details are local to this workflow:

- Panels are sized from the subplot slot (`_prepare_panel`), not from
  `ax.set_aspect(..., adjustable="datalim")` — the latter silently crops
  parcels that sit near the frame edge.
- Map labels are field indices only (`_place_labels`); acres live in A1 and the
  geometry table so the boxes stay small enough to place without collisions.

## Common tasks

### Compare one grower against another

```bash
... run_field_eda.py --growers il,ia --output-dir /tmp/il-vs-ia
```

Then read `tables/fields_geometry.csv` and `comparisons.json`; the figures
regenerate at the smaller n.

### Reuse the tables elsewhere

```python
import pandas as pd
geo = pd.read_csv(f"{out}/tables/fields_geometry.csv")
weather = pd.read_csv(f"{out}/tables/weather_field_year.csv")
```

### Check the numbers behind a figure

`comparisons.json` stores every statistic that a figure prints, including the
nested variance shares and the per-year Spearman values.

## Caveats

- Boundaries are OSM/CSB parcels; positional accuracy is not ground truth.
- CDL is 30 m and annual: a parcel edge mixes classes, so sub-95 % coverage is
  an edge effect, not a data error.
- NASA POWER cells are ~50 km wide; a farm's "field-level" weather is one or
  two values, and identical series are grid resolution, not microclimate.
- One demo field (IA-06) has no CDL rows; CDL analyses run on 29 of 30 fields.

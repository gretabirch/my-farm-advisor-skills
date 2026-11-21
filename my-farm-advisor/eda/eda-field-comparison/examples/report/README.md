# Field-level EDA report (example artifact)

`field-eda-report.md` is a **one-time assembly artifact**, not something the
workflow generates. The reusable subskill stops at `figures/`, `tables/`,
`comparisons.json` and `manifest.json`; report prose lives here so the subskill
stays report-free.

## What is in here

| Path | Content |
|---|---|
| `field-eda-summary.md` | one-page version: three sections, three figures, headline numbers only |
| `field-eda-report.md` | the assembled report: three categories, four comparison levels, numbers quoted from `comparisons.json` |
| `figures/*.jpg` | six of the twelve EDA figures plus three soil-texture panels, 1200 px, re-encoded from the run output (≈1.1 MB total) |

## How it was produced

1. Run the workflow (see the report's last section) against the demo runtime.
2. Read `manifest.json` for figure titles, stories, caveats and levels, and
   `comparisons.json` for the statistics.
3. Downscale six figures to 1200 px wide and write the prose by hand.

No runtime tables, GeoJSON, or full-resolution figures are committed; they stay
under `${DATA_PIPELINE_DATA_ROOT}/data-pipeline/eda/<run>/`.

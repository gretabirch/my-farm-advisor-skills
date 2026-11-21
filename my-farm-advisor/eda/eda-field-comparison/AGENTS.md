# Local Instructions

## Purpose

This folder owns field-level EDA across growers: parcel geometry, CDL land
cover and rotation, and growing-season weather. It produces static figures,
CSV tables, a comparisons JSON, and a manifest — no dashboard, no report
generation, no soil layers.

## Safe edit scope

Edits stay inside `eda-field-comparison/`. Do not add a `SKILL.md` here, do not
change `../../SKILL.md` identity, and do not edit sibling EDA workflows.

## Read nearby docs first

Read `GUIDE.md` for the figure catalog, output contract, and statistics. For
routing context read `../INDEX.md` and `../../SKILL.md`. For map-rendering
rules read `../../data-pipeline/src/scripts/README.md` and the field-frame
helpers under `../../data-pipeline/src/scripts/lib/`.

## Local conventions

- `src/run_field_eda.py` is the only entrypoint: input gate → stats → figures
  → manifest. `--skip-maps` and `--figures A1,C3` exist for iteration.
- Story, caveat, and comparison level for each figure live in the `FIGURES`
  table in `run_field_eda.py`. Figures quote the text through
  `manifest.json`; keep the two in sync when a claim changes.
- Never commit runtime artefacts. Render to
  `${DATA_PIPELINE_DATA_ROOT}/data-pipeline/eda/<run-name>/`, which holds
  `figures/`, `tables/`, `comparisons.json`, and `manifest.json`.
- Only the three demo PNGs in `examples/output/` are committed.
- Renderers take a `ctx` dict; add a new figure by adding a renderer to
  `eda_figures.py` and an entry to `FIGURES`, not by editing the CLI.
- Keep every statistical claim descriptive. With three growers × ten fields,
  p-values illustrate effect size; they never license inference.

## Figure gate

Before committing a new or regenerated figure, open the PNG and check: no
empty or under-covered panel, local-metre axes with a boxed north arrow on
every farm panel, halo/edge-legend placement, categorical CDL colours versus
continuous weather colours, and legible labels at README size. A change to
`_prepare_panel` or `_place_labels` re-triggers review of all three maps.

## Local validation

Run `./scripts/validate.sh` from the repository root after structural changes.
For a quick behavioural check, run the workflow with `--skip-maps` against the
demo runtime and confirm twelve `ok` lines on a full run.

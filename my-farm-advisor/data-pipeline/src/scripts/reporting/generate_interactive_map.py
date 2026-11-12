#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false
"""Generate the standalone interactive Leaflet map for the canonical farm.

The renderer itself lives in the ``interactive-web-map`` skill; this wrapper
only resolves the skill path, points it at the runtime farm tree, and writes
the map beside the HTML/Markdown farm reports so the report can link to it.

The map is a pure function of the farm tree, renders in well under a second,
and caches Leaflet under ``shared/vendor``, so it is always rebuilt rather than
skipped by a staleness manifest.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_LOCAL_LIB = Path(__file__).resolve().parents[1] / "lib"
sys.path.insert(0, str(_LOCAL_LIB))

from runtime_paths import resolve_runtime_paths  # noqa: E402

_RUNTIME_PATHS = resolve_runtime_paths()
_REPO = _RUNTIME_PATHS.runtime_base
_SCRIPTS = _RUNTIME_PATHS.runtime_scripts
_LIB = _RUNTIME_PATHS.runtime_scripts / "lib"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_LIB))

from reporting_bootstrap import ensure_skill_path  # noqa: E402

_MAP_SKILL = ensure_skill_path("interactive-web-map")
sys.path.insert(0, str(_MAP_SKILL))

import render_farm_map  # noqa: E402

from paths import farm_interactive_map_path  # noqa: E402

_DEFAULT_GROWER = os.environ.get("AG_GROWER_SLUG", "default-grower")
_DEFAULT_FARM = os.environ.get("AG_FARM_SLUG", "default-farm")


def main() -> None:
    print("=" * 60)
    print("Interactive field map — standalone Leaflet dashboard")
    print("=" * 60)

    grower_slug = _DEFAULT_GROWER
    farm_slug = _DEFAULT_FARM
    farm_dir = _REPO / "growers" / grower_slug / "farms" / farm_slug
    if not farm_dir.exists():
        raise SystemExit(f"Farm directory not found: {farm_dir}")

    output_path = farm_interactive_map_path(grower_slug, farm_slug)
    assets_dir = _REPO / "shared" / "vendor"

    payload = render_farm_map.build_payload(farm_dir, grower_slug, farm_slug)
    leaflet_js, leaflet_css = render_farm_map.leaflet_assets(assets_dir)
    html = render_farm_map.build_html(payload, leaflet_js, leaflet_css)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    print(f"✓ Interactive map saved → {output_path}")
    print(
        "  "
        + json.dumps(
            {
                "fields": payload["meta"]["field_count"],
                "acres": payload["meta"]["total_acres"],
                "leaflet_inlined": bool(leaflet_js and leaflet_css),
                "size_kb": round(output_path.stat().st_size / 1024, 1),
            }
        )
    )


if __name__ == "__main__":
    main()

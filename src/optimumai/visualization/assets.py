"""Access the compiled **OptiX** widget bundle (TypeScript → esbuild IIFE).

OptiX is authored in TypeScript under ``web/`` (typed, unit-tested with Vitest)
and compiled by esbuild to a single self-contained IIFE at
``visualization/_static/optix.js`` — committed to the repo and shipped in the
wheel. Python inlines that bundle into generated HTML so every widget stays
**offline and CDN-free**; node is only needed at dev/build time, never at
install or runtime.

    from optimumai.visualization.assets import optix_js
    html = f"<script>{optix_js()}</script>"   # defines window.OptiX
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files


@lru_cache(maxsize=1)
def optix_js() -> str:
    """Return the OptiX bundle source (``window.OptiX`` IIFE).

    Raises ``FileNotFoundError`` if the compiled asset wasn't shipped — build it
    with ``cd web && npm install && npm run build``.
    """
    asset = files("optimumai.visualization").joinpath("_static", "optix.js")
    return asset.read_text(encoding="utf-8")


def optix_script_tag() -> str:
    """The OptiX bundle wrapped in an inline ``<script>`` tag."""
    return f"<script>{optix_js()}</script>"

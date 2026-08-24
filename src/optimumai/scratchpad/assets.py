"""Front-end assets: CDN by default, vendored locally on request.

The scratchpad is described as local-first, and for the *server* that is true --
it binds 127.0.0.1 and makes no API calls. But the page pulled JSXGraph and
KaTeX from a CDN, so the first load of a board needed the network. On a plane
the board simply did not draw.

``vendor_assets()`` downloads each file once into ``static/vendor/``;
``resolve_assets()`` then prefers those copies. It also only returns the
libraries a given board actually declares, so a vectors board does not download
or load a maths typesetter it never uses.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

CDN_TIMEOUT_S = 30


@dataclass(frozen=True)
class Asset:
    key: str
    url: str
    filename: str


#: Every third-party file the scratchpad can load, pinned by version.
ASSETS: tuple[Asset, ...] = (
    Asset("jsxgraph_css", "https://cdn.jsdelivr.net/npm/jsxgraph@1.11.1/distrib/jsxgraph.css",
          "jsxgraph.css"),
    Asset("jsxgraph_js", "https://cdn.jsdelivr.net/npm/jsxgraph@1.11.1/distrib/jsxgraphcore.js",
          "jsxgraphcore.js"),
    Asset("katex_css", "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css",
          "katex.min.css"),
    Asset("katex_js", "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js",
          "katex.min.js"),
)

ASSETS_BY_KEY = {a.key: a for a in ASSETS}

#: Which asset keys each board *library* needs.
LIBRARY_ASSETS: dict[str, tuple[str, ...]] = {
    "jsxgraph": ("jsxgraph_css", "jsxgraph_js"),
}
KATEX_ASSETS: tuple[str, ...] = ("katex_css", "katex_js")


def vendor_dir(package_root: Path | None = None) -> Path:
    root = package_root or Path(__file__).resolve().parent
    return root / "static" / "vendor"


def required_keys(library: str, needs_katex: bool) -> list[str]:
    """Asset keys for one board -- nothing more."""
    keys = list(LIBRARY_ASSETS.get(library, ()))
    if needs_katex:
        keys.extend(KATEX_ASSETS)
    return keys


def is_vendored(keys: list[str], directory: Path | None = None) -> bool:
    """True when every required file is already on disk."""
    d = directory or vendor_dir()
    return bool(keys) and all((d / ASSETS_BY_KEY[k].filename).is_file() for k in keys)


def resolve_assets(
    library: str, needs_katex: bool, directory: Path | None = None
) -> dict[str, str | bool]:
    """Map asset keys to URLs for this board, preferring vendored copies.

    Keys a board does not need are absent, so the template emits no tag for
    them at all.
    """
    keys = required_keys(library, needs_katex)
    offline = is_vendored(keys, directory)
    out: dict[str, str | bool] = {"offline": offline}
    for key in keys:
        asset = ASSETS_BY_KEY[key]
        out[key] = f"/static/vendor/{asset.filename}" if offline else asset.url
    return out


def vendor_assets(directory: Path | None = None) -> list[tuple[str, str]]:
    """Download every asset into ``directory``. Returns ``(filename, status)``.

    Status is ``"ok"``, ``"cached"``, or an error string -- one failure does not
    abort the rest, so a partial network gets you as far as it can.
    """
    d = directory or vendor_dir()
    d.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, str]] = []
    for asset in ASSETS:
        target = d / asset.filename
        if target.is_file() and target.stat().st_size > 0:
            results.append((asset.filename, "cached"))
            continue
        try:
            with urllib.request.urlopen(asset.url, timeout=CDN_TIMEOUT_S) as resp:
                target.write_bytes(resp.read())
            results.append((asset.filename, "ok"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            results.append((asset.filename, f"failed: {exc}"))
    return results

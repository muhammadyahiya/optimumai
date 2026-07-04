"""The OptiX bundle ships and the neural-net playground is self-contained.

The OptiX widget kit is authored + unit-tested in TypeScript under ``web/`` (see
``web/test/*.test.ts``); these tests just assert the compiled bundle is present in
the installed package and that the Python ``nn_playground`` embeds it as offline,
CDN-free HTML.
"""

from __future__ import annotations

import pytest

from optimumai.visualization.assets import optix_js, optix_script_tag
from optimumai.visualization.playgrounds import nn_playground, playground


def test_optix_bundle_ships_and_is_an_iife():
    js = optix_js()
    assert len(js) > 2000
    assert "OptiX" in js
    # esbuild IIFE, not a module — safe to inline in a <script> tag.
    assert "require(" not in js
    assert optix_script_tag().startswith("<script>")


def test_nn_playground_is_self_contained(tmp_path):
    out = str(tmp_path / "nn.html")
    path = nn_playground(out=out)
    html = open(path, encoding="utf-8").read()
    assert "OptiX.mount.nnPlayground" in html
    assert "optix-nn" in html
    # offline: no external scripts, no CDN.
    assert "src=" not in html
    assert "http://" not in html and "https://" not in html


def test_dispatcher_routes_nn(tmp_path):
    out = str(tmp_path / "nn.html")
    assert playground("nn", out=out) == out


def test_dispatcher_rejects_unknown():
    with pytest.raises(ValueError):
        playground("nope")

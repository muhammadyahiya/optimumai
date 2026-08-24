"""Sprint 1 — design tokens + the guided-step explainer UI."""

from __future__ import annotations

import pytest

from optimumai.design import MOTION, PALETTE, to_ansi, to_css_vars
from optimumai.visualization.explain import (
    CONCEPTS,
    _build_html,
    _steps,
    list_explain_concepts,
)

# concepts authored with the guided-step fields in Sprint 1
GUIDED_CONCEPTS = ["attention", "backpropagation", "gradient_descent"]


# --- design tokens --------------------------------------------------------


def test_palette_reserves_attention_colour():
    """Yellow is the one attention channel; it must not double as a semantic role."""
    semantic = [
        PALETTE.basis_x, PALETTE.basis_y, PALETTE.basis_z,
        PALETTE.given, PALETTE.derivative, PALETTE.result,
        PALETTE.pair_second, PALETTE.counterexample,
        PALETTE.hint, PALETTE.justification,
    ]
    assert PALETTE.attention not in semantic


def test_signed_parameter_ramps_are_distinct_hue_families():
    """Sign selects the hue family; magnitude selects lightness within it."""
    assert PALETTE.param_pos_dark != PALETTE.param_neg_dark
    assert PALETTE.param_pos_light != PALETTE.param_neg_light
    # data must be greyscale so it can never be mistaken for a learned parameter
    assert PALETTE.data_dark not in (PALETTE.param_pos_dark, PALETTE.param_neg_dark)


def test_motion_orders_durations_by_how_much_moves():
    """A deforming continuum needs longer than a positional move."""
    assert MOTION.move_ms < MOTION.morph_ms < MOTION.deform_ms
    assert MOTION.ui_fast_ms < MOTION.ui_base_ms < MOTION.move_ms


def test_to_css_vars_emits_custom_properties():
    css = to_css_vars()
    assert "--attention:" in css
    assert "--basis-x:" in css
    assert "--ease-css:" in css
    assert "--ui-base-ms: 200ms;" in css
    # no python identifiers leaked through
    assert "_notes" not in css


def test_to_ansi_returns_empty_for_unknown_role():
    assert to_ansi("hint") != ""
    assert to_ansi("not_a_role") == ""


# --- step model ----------------------------------------------------------


def test_steps_defaults_are_backwards_compatible():
    """A step authored before Sprint 1 must still build, with empty new fields."""
    (step,) = _steps({"title": "Do the thing", "narration": "n"})
    assert step["hint"] is None
    assert step["justification"] is None
    assert step["substeps"] == []


def test_steps_normalises_substeps():
    (step,) = _steps(
        {
            "title": "T", "narration": "n",
            "hint": "h", "justification": "j",
            "substeps": [{"title": "a", "detail": "d"}, {"title": "b", "formula": "x"}],
        }
    )
    assert step["hint"] == "h"
    assert step["justification"] == "j"
    assert [s["title"] for s in step["substeps"]] == ["a", "b"]
    # every substep has the full key set, whether or not it was authored
    assert step["substeps"][0]["formula"] is None
    assert step["substeps"][1]["detail"] == ""


def test_every_concept_still_builds():
    """All 30 pre-existing concepts must survive the schema change."""
    assert len(list_explain_concepts()) >= 30
    for name in list_explain_concepts():
        for step in CONCEPTS[name]["steps"]:
            assert {"hint", "justification", "substeps"} <= set(step)


@pytest.mark.parametrize("concept", GUIDED_CONCEPTS)
def test_guided_concepts_have_hint_and_justification_on_every_step(concept):
    for step in CONCEPTS[concept]["steps"]:
        assert step["hint"], f"{concept} step {step['index']} has no hint"
        assert step["justification"], f"{concept} step {step['index']} has no justification"


@pytest.mark.parametrize("concept", GUIDED_CONCEPTS)
def test_guided_concepts_respect_the_granularity_rule(concept):
    """Top level is strategy: keep it to 3-8 steps and push length into substeps."""
    steps = CONCEPTS[concept]["steps"]
    assert 3 <= len(steps) <= 8, f"{concept} has {len(steps)} top-level steps"
    assert any(step["substeps"] for step in steps), f"{concept} has no level-2 detail"


@pytest.mark.parametrize("concept", GUIDED_CONCEPTS)
def test_hint_does_not_simply_restate_the_narration(concept):
    """A hint names the technique; it must not just be the answer again."""
    for step in CONCEPTS[concept]["steps"]:
        assert step["hint"] != step["narration"]


# --- rendered page -------------------------------------------------------


def test_build_html_substitutes_design_tokens():
    html = _build_html("attention")
    assert "__CSS_VARS__" not in html
    assert "--attention:" in html
    assert PALETTE.hint in html


def test_build_html_has_no_unsubstituted_placeholders():
    html = _build_html("attention")
    assert "__TRACE_JSON__" not in html
    assert "__TITLE__" not in html


def test_build_html_contains_guided_step_controls():
    html = _build_html("attention")
    for hook in ('id="hint-block"', 'id="why-block"', 'id="substeps"',
                 'id="showall"', 'id="hints-toggle"', 'id="restart"'):
        assert hook in html, f"missing {hook}"


def test_build_html_encodes_the_index_phase_cursor():
    """Advancing from a hint must reveal that step, not skip to the next one."""
    html = _build_html("attention")
    assert "function inHintPhase()" in html
    assert 'phase = "revealed"; render(); return;' in html
    assert "location.hash.match(/step=([0-9]+)/)" in html


def test_build_html_carries_authored_hints_into_the_payload():
    html = _build_html("gradient_descent")
    assert "Before you can go downhill you need to know how high you are" in html


def test_concept_without_hints_still_renders():
    html = _build_html("softmax")
    assert 'id="hint-block"' in html
    assert '"hint": null' in html or '"hint":null' in html

"""Color and motion tokens, derived from Manim's published conventions.

The raw ladders are ManimGL's (``manimlib/default_config.yml``) — i.e. the
values used in the 3Blue1Brown videos — rather than Manim Community's, which
forked two of them. Two conventions from that source are load-bearing and are
encoded here as *semantic* names rather than left to each caller:

* ``attention`` (yellow) is reserved for "look here" and is never given a
  semantic role, because every Manim indication animation defaults to it.
* Sign selects a *hue family* and magnitude selects *lightness within it*, so
  signed values stay readable either side of zero. Learned parameters use the
  blue/red families; data and activations use greyscale — which is what lets a
  reader tell a weight from an activation at a glance.

Note on matching published video frames: 3Blue1Brown's render config applies an
ffmpeg ``saturation=1.5`` filter, so sampling a published frame yields colors
that do not exist in the library. These are the pre-filter source values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# --------------------------------------------------------------------------
# raw ladders (_A lightest -> _E darkest; the bare name aliases _C)
# --------------------------------------------------------------------------

BLUE_A, BLUE_B, BLUE_C, BLUE_D, BLUE_E = "#C7E9F1", "#9CDCEB", "#58C4DD", "#29ABCA", "#1C758A"
TEAL_A, TEAL_B, TEAL_C, TEAL_D, TEAL_E = "#ACEAD7", "#76DDC0", "#5CD0B3", "#55C1A7", "#49A88F"
GREEN_A, GREEN_B, GREEN_C, GREEN_D, GREEN_E = "#C9E2AE", "#A6CF8C", "#83C167", "#77B05D", "#699C52"
YELLOW_A, YELLOW_B, YELLOW_C, YELLOW_D, YELLOW_E = "#FFF1B6", "#FFEA94", "#FFFF00", "#F4D345", "#E8C11C"  # noqa: E501
RED_A, RED_B, RED_C, RED_D, RED_E = "#F7A1A3", "#FF8080", "#FC6255", "#E65A4C", "#CF5044"
MAROON_A, MAROON_B, MAROON_C = "#ECABC1", "#EC92AB", "#C55F73"
PURPLE_A, PURPLE_B, PURPLE_C = "#CAA3E8", "#B189C6", "#9A72AC"
GREY_A, GREY_B, GREY_C, GREY_D, GREY_E = "#DDDDDD", "#BBBBBB", "#888888", "#444444", "#222222"

WHITE, BLACK = "#FFFFFF", "#000000"
PINK, ORANGE = "#D147BD", "#FF862F"

#: Grant Sanderson's heatmap gradient (``COLORMAP_3B1B``).
COLORMAP_3B1B = (BLUE_E, GREEN_C, YELLOW_C, RED_C)


@dataclass(frozen=True)
class Palette:
    """Semantic color roles. Assign once here; never pick a hex at a call site."""

    # page chrome
    canvas: str = "#0F1117"
    surface: str = "#171A23"
    recessed: str = GREY_E
    border: str = "#2A2F3D"
    text: str = "#E7E9EE"
    text_dim: str = "#7D8497"

    # the one attention channel -- reserved, never semantic
    attention: str = YELLOW_D

    # basis directions (and, by extension, matrix columns)
    basis_x: str = GREEN_C
    basis_y: str = RED_C
    basis_z: str = BLUE_D

    # coordinate space
    grid: str = BLUE_D
    grid_faded: str = GREY_C

    # signed learned parameters: hue family = sign, lightness = magnitude
    param_pos_dark: str = BLUE_E
    param_pos_light: str = BLUE_B
    param_neg_dark: str = RED_E
    param_neg_light: str = RED_B

    # data / activations -- greyscale so they can never read as parameters
    data_dark: str = GREY_C
    data_light: str = WHITE

    # roles in a derivation
    given: str = BLUE_C
    derivative: str = GREEN_C
    result: str = TEAL_C
    pair_second: str = MAROON_B
    counterexample: str = PINK

    # step-panel semantics (three distinct epistemic statuses, three weights)
    hint: str = TEAL_C
    justification: str = PURPLE_A
    warning: str = ORANGE

    #: Opacity for a faded "where it was before" reference layer.
    ghost_opacity: float = 0.3
    #: Opacity for non-focal items when one item is emphasised.
    dim_opacity: float = 0.2


@dataclass(frozen=True)
class Motion:
    """Timing contract.

    Two scales, deliberately separate. ``*_ms`` values animate *mathematics*
    and come from Manim's defaults; ``ui_*_ms`` values animate *affordances*
    (a panel opening) where those durations would feel broken.
    """

    #: Positional move of an existing object.
    move_ms: int = 1000
    #: Symbolic rewrite where glyph identity must be tracked.
    morph_ms: int = 2000
    #: Deformation of a continuum (a grid warping under a transformation).
    deform_ms: int = 3000
    #: The pause a narration lands in, between animations.
    beat_ms: int = 1000

    ui_fast_ms: int = 120
    ui_base_ms: int = 200

    #: Stagger for an enumerable collection. A continuum must use 0.0.
    lag_ratio_group: float = 0.05
    #: Upper bound on write-on stagger; heavy overlap reads as handwriting.
    lag_ratio_write_max: float = 0.2

    #: Closest CSS approximation to quintic smootherstep.
    ease_css: str = "cubic-bezier(0.65, 0, 0.35, 1)"
    #: The exact curve, for JS/canvas use: 6t^5 - 15t^4 + 10t^3.
    ease_js: str = "t*t*t*(t*(6*t-15)+10)"

    _notes: tuple[str, ...] = field(
        default=(
            "Stagger enumerable things; never stagger a continuum.",
            "When a staged reveal supplies the envelope, keep per-element easing linear.",
        ),
        repr=False,
    )


PALETTE = Palette()
MOTION = Motion()


def _css_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def to_css_vars(palette: Palette | None = None, motion: Motion | None = None) -> str:
    """Render the tokens as CSS custom properties for a ``:root`` block.

    Returns the declarations only (no selector), so a caller can drop them into
    whatever wrapper it already emits.
    """
    pal = palette or PALETTE
    mot = motion or MOTION
    lines = [f"  {_css_name(k)}: {v};" for k, v in asdict(pal).items() if isinstance(v, str)]
    lines += [
        f"  {_css_name(k)}: {v};"
        for k, v in asdict(pal).items()
        if isinstance(v, float)
    ]
    for key, val in asdict(mot).items():
        if key.startswith("_"):
            continue
        suffix = "ms" if key.endswith("_ms") else ""
        lines.append(f"  {_css_name(key)}: {val}{suffix};")
    return "\n".join(lines)


_ANSI = {
    "text": "\033[0m",
    "text_dim": "\033[2m",
    "attention": "\033[33m",
    "hint": "\033[36m",
    "justification": "\033[35m",
    "warning": "\033[33m",
    "result": "\033[36m",
    "given": "\033[34m",
    "derivative": "\033[32m",
    "counterexample": "\033[35m",
}
RESET = "\033[0m"


def to_ansi(role: str) -> str:
    """ANSI escape for a palette *role*, or an empty string if it has no mapping.

    Deliberately a small 8-color mapping rather than truecolor: it degrades
    correctly on basic terminals, and callers that honour ``NO_COLOR`` can just
    skip calling this.
    """
    return _ANSI.get(role, "")

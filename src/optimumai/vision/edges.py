"""Sobel edge detection — finding boundaries with two fixed convolutions.

Intuition
---------
An edge is a place where brightness changes sharply. The Sobel operator finds
edges by convolving the image with two small, hand-designed kernels: one that
responds to horizontal brightness change (``Gx``, a vertical edge detector)
and one that responds to vertical brightness change (``Gy``, a horizontal
edge detector). Combining the two responses at each pixel gives both *how
strong* the edge is and *which direction* it runs.

Math
----
The two 3×3 Sobel kernels approximate the image's partial derivatives:

    Gx = [[-1, 0, 1],      Gy = [[-1, -2, -1],
          [-2, 0, 2],            [ 0,  0,  0],
          [-1, 0, 1]]            [ 1,  2,  1]]

``Gx`` is large where intensity changes left-to-right (a vertical edge);
``Gy`` is large where intensity changes top-to-bottom (a horizontal edge).
Convolving the image with each kernel gives two gradient images, then per
pixel:

    magnitude  = √(Gx² + Gy²)          — edge strength, direction-agnostic
    orientation = atan2(Gy, Gx)        — the edge's direction, in radians

Why AI uses it
---------------
* Sobel is a **fixed, hand-designed** convolution — exactly the operation a
  CNN's first layer *learns* automatically. Visualizing what a trained
  CNN's early filters converge to almost always turns up edge- and
  gradient-like kernels resembling Sobel/Gabor filters; this module shows
  the mechanism a CNN would otherwise have to discover from data.
* Edges are the first rung of the classic vision hierarchy: edges → textures
  → object parts → objects. Every deeper CNN feature builds on detecting
  boundaries like these.
* Gradient magnitude and orientation are the backbone of classical descriptors
  (HOG, SIFT) used for detection before deep learning, and remain useful
  hand-crafted features and pre-processing/augmentation signals today.
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.trace import Trace
from optimumai.vision.convolution import conv2d

GX = np.array(
    [
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ]
)
GY = np.array(
    [
        [-1.0, -2.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
    ]
)


def sobel_edges(x, padding: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(magnitude, orientation)`` gradient maps for image ``x``.

    ``orientation`` is in radians, from ``atan2(Gy, Gx)`` (range ``(-π, π]``).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"sobel_edges expects a 2-D image, got shape {x.shape}")
    if x.shape[0] < 3 or x.shape[1] < 3:
        raise ValueError(f"sobel_edges needs at least a 3x3 image, got shape {x.shape}")
    gx = conv2d(x, GX, stride=1, padding=padding)
    gy = conv2d(x, GY, stride=1, padding=padding)
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx)
    return magnitude, orientation


def sobel_edges_trace(x, padding: int = 1) -> Trace:
    """Build the full trace of Sobel edge detection: Gx, Gy, magnitude, orientation."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"sobel_edges_trace expects a 2-D image, got shape {x.shape}")
    if x.shape[0] < 3 or x.shape[1] < 3:
        raise ValueError(f"sobel_edges_trace needs at least a 3x3 image, got shape {x.shape}")

    gx = conv2d(x, GX, stride=1, padding=padding)
    gy = conv2d(x, GY, stride=1, padding=padding)
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx)
    peak_idx = tuple(int(i) for i in np.unravel_index(np.argmax(magnitude), magnitude.shape))
    peak_val = float(magnitude[peak_idx])

    t = Trace(
        op="sobel_edges",
        formula="Gx = x * Kx;  Gy = x * Ky;  magnitude = √(Gx² + Gy²);  "
        "orientation = atan2(Gy, Gx)",
        complexity=f"two O(H·W·9) convolutions + O(H·W) elementwise combine, "
        f"output {magnitude.shape}",
        why_ai=[
            "Sobel is a fixed convolution doing exactly what a CNN's first layer "
            "learns to do from data: respond to local brightness gradients",
            "Trained CNNs' first-layer filters routinely converge to edge/gradient "
            "detectors that look like Sobel or Gabor kernels",
            "Edges are rung one of the vision feature hierarchy: edges → textures "
            "→ parts → objects",
            "Gradient magnitude + orientation underlie classical descriptors "
            "(HOG, SIFT, Canny) used for detection before end-to-end deep nets",
        ],
        meta={
            "input_shape": x.shape,
            "padding": padding,
            "output_shape": magnitude.shape,
            "peak_position": peak_idx,
            "peak_magnitude": peak_val,
        },
    )

    t.add("Gx kernel (vertical-edge detector)", f"Kx =\n{arr(GX)}", GX)
    t.add(
        "Gx = x ⋆ Kx (horizontal gradient)",
        f"Gx =\n{arr(gx)}",
        gx,
        detail="Large |Gx| marks a vertical edge — intensity changing left-to-right.",
    )
    t.add("Gy kernel (horizontal-edge detector)", f"Ky =\n{arr(GY)}", GY)
    t.add(
        "Gy = x ⋆ Ky (vertical gradient)",
        f"Gy =\n{arr(gy)}",
        gy,
        detail="Large |Gy| marks a horizontal edge — intensity changing top-to-bottom.",
    )
    t.add(
        "Gradient magnitude",
        f"√(Gx² + Gy²) =\n{arr(magnitude)}",
        magnitude,
        detail=f"Strongest edge response is {num(peak_val)} at output position {peak_idx}.",
    )
    t.add(
        "Gradient orientation",
        f"atan2(Gy, Gx) [radians] =\n{arr(orientation)}",
        orientation,
        detail="0 points along +x (rightward gradient); ±π/2 points along ±y.",
    )

    t.result = magnitude
    return t


def demo(seed: int = 0) -> Trace:
    """Find the edge in a 6×6 image with a sharp dark-to-bright vertical boundary."""
    x = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ]
    )
    return sobel_edges_trace(x, padding=1)

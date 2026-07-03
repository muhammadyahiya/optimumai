"""A tiny CNN forward pass — watching tensor shapes flow through a network.

Intuition
---------
A convolutional classifier is a pipeline: each stage transforms the tensor's
*shape* as much as its values. Understanding a CNN starts with understanding
that shape pipeline before worrying about what any specific number means:

    image (H×W) → conv → feature map → ReLU → max-pool → flatten (1-D)
                → dense (weighted sum) → softmax → class probabilities

This module wires together :func:`~optimumai.vision.convolution.conv2d` and
:func:`~optimumai.vision.pooling.max_pool2d` — the same functions from the
other vision modules — with two new, minimal pieces (ReLU, a dense layer) and
a numerically stable softmax, then runs one small single-channel image all
the way through to a class distribution.

Math
----
Five stages, in order:

1. **Convolution**: ``feature_map = conv2d(image, kernel)``, shape
   ``(H − k + 1, W − k + 1)`` with stride 1, no padding.
2. **ReLU**: ``relu(z) = max(0, z)``, elementwise — a nonlinearity that costs
   almost nothing and does not shrink or reshape the tensor.
3. **Max pooling**: ``max_pool2d(relu_map, kernel_size, stride)`` shrinks the
   spatial size by ``stride`` while keeping the strongest activations.
4. **Flatten**: reshape the 2-D pooled map into a 1-D vector of length
   ``pooled_h · pooled_w`` — the bridge from "spatial" to "just a list of
   numbers a dense layer can read."
5. **Dense + softmax**: ``logits = W @ flat + b`` (one number per class),
   then ``softmax(logits)ᵢ = e^(logits_i − max) / Σⱼ e^(logits_j − max)``
   turns the logits into a probability distribution over classes.

Why AI uses it
---------------
* This exact stack (conv → nonlinearity → pool → ... → dense → softmax) is
  LeNet-5 (1998) in miniature, and the same skeleton — with far more layers,
  channels, and tricks — underlies AlexNet, VGG, ResNet, and the vision
  backbones inside modern multimodal models.
* Watching the **shape** at each stage is the practical skill: a real
  network is "just" a sequence of shape transformations, and the most common
  bugs (dimension mismatches) are caught by tracking shapes, not values.
* ReLU is the nonlinearity that lets stacked convolutions do more than a
  single big linear filter could; pooling and flattening are exactly the
  glue functions from :mod:`optimumai.vision.pooling` reused here, not new
  math; softmax closes the loop back to
  :mod:`optimumai.probability.softmax`.
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.trace import Trace
from optimumai.vision.convolution import conv2d
from optimumai.vision.pooling import max_pool2d


def relu(z) -> np.ndarray:
    """Elementwise ``max(0, z)`` — zero out negative activations, keep the rest."""
    return np.maximum(0.0, np.asarray(z, dtype=float))


def dense(flat, weights, bias) -> np.ndarray:
    """A fully-connected layer: ``logits = weights @ flat + bias``."""
    flat = np.asarray(flat, dtype=float)
    weights = np.asarray(weights, dtype=float)
    bias = np.asarray(bias, dtype=float)
    if weights.ndim != 2 or weights.shape[1] != flat.shape[0]:
        raise ValueError(
            f"weights must have shape (n_classes, {flat.shape[0]}), got {weights.shape}"
        )
    if bias.shape != (weights.shape[0],):
        raise ValueError(f"bias must have shape ({weights.shape[0]},), got {bias.shape}")
    return weights @ flat + bias


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def cnn_forward(
    image, kernel, dense_weights, dense_bias, pool_size: int = 2
) -> np.ndarray:
    """Run one image through conv -> ReLU -> max-pool -> flatten -> dense -> softmax."""
    feature_map = conv2d(image, kernel, stride=1, padding=0)
    activated = relu(feature_map)
    pooled = max_pool2d(activated, kernel_size=pool_size, stride=pool_size)
    flat = pooled.flatten()
    logits = dense(flat, dense_weights, dense_bias)
    return _softmax(logits)


def cnn_forward_trace(
    image, kernel, dense_weights, dense_bias, pool_size: int = 2
) -> Trace:
    """Build the full trace of a tiny CNN forward pass, headlined by shape flow."""
    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    dense_weights = np.asarray(dense_weights, dtype=float)
    dense_bias = np.asarray(dense_bias, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"cnn_forward_trace expects a 2-D single-channel image, got {image.shape}")

    feature_map = conv2d(image, kernel, stride=1, padding=0)
    activated = relu(feature_map)
    pooled = max_pool2d(activated, kernel_size=pool_size, stride=pool_size)
    flat = pooled.flatten()
    logits = dense(flat, dense_weights, dense_bias)
    probs = _softmax(logits)

    shape_flow = (
        f"{image.shape} --conv--> {feature_map.shape} --relu--> {activated.shape} "
        f"--pool--> {pooled.shape} --flatten--> {flat.shape} --dense--> {logits.shape} "
        f"--softmax--> {probs.shape}"
    )

    t = Trace(
        op="cnn_forward",
        formula="softmax(W · flatten(maxpool(relu(x ⋆ kernel))) + b)",
        complexity=f"conv O(H·W·k²) + pool O(H'·W') + dense O(n_classes·flat_len); "
        f"n_classes={dense_weights.shape[0]}",
        why_ai=[
            "This is LeNet-5's skeleton in miniature: conv -> nonlinearity -> pool "
            "-> ... -> dense -> softmax underlies AlexNet, VGG, ResNet and beyond",
            "Tracking the SHAPE at each stage is the practical skill — a real "
            "network is a sequence of shape transformations, and dimension "
            "mismatches are the most common real-world bug",
            "ReLU is what lets stacking convolutions learn more than one big "
            "linear filter ever could; without it, conv -> conv collapses to conv",
            "Flatten is the bridge from spatial (2-D) to a plain vector a dense "
            "layer can read; softmax closes the loop back to a class distribution",
        ],
        meta={
            "shape_flow": shape_flow,
            "image_shape": image.shape,
            "feature_map_shape": feature_map.shape,
            "pooled_shape": pooled.shape,
            "flat_len": flat.shape[0],
            "n_classes": dense_weights.shape[0],
            "predicted_class": int(np.argmax(probs)),
        },
    )

    t.add(
        "Input image",
        f"x, shape {image.shape} =\n{arr(image)}",
        image,
    )
    t.add(
        f"Conv2d: {image.shape} -> {feature_map.shape}",
        f"feature_map = x ⋆ kernel (kernel shape {kernel.shape}) =\n{arr(feature_map)}",
        feature_map,
        detail="See optimumai.vision.convolution for the window-by-window trace.",
    )
    t.add(
        f"ReLU: {feature_map.shape} -> {activated.shape} (shape unchanged)",
        f"max(0, feature_map) =\n{arr(activated)}",
        activated,
        detail="ReLU never changes shape — only zeros out negative activations.",
    )
    t.add(
        f"Max-pool: {activated.shape} -> {pooled.shape}",
        f"max_pool2d(activated, size={pool_size}) =\n{arr(pooled)}",
        pooled,
        detail="See optimumai.vision.pooling for the window-by-window trace.",
    )
    t.add(
        f"Flatten: {pooled.shape} -> {flat.shape}",
        f"flat = pooled.reshape(-1) =\n{arr(flat)}",
        flat,
        detail="Row-major flatten: spatial layout becomes a plain ordered list.",
    )
    t.add(
        f"Dense: {flat.shape} -> {logits.shape}",
        f"logits = W @ flat + b, W shape {dense_weights.shape} =\n{arr(logits)}",
        logits,
        detail="One logit per class; sign and magnitude do not yet sum to anything meaningful.",
    )
    t.add(
        f"Softmax: {logits.shape} -> {probs.shape} (class distribution)",
        f"softmax(logits) =\n{arr(probs)}",
        probs,
        detail=f"Predicted class = argmax = {int(np.argmax(probs))}; "
        f"probabilities sum to {num(float(np.sum(probs)))}.",
    )
    t.add("Shape flow, start to finish", shape_flow, shape_flow)

    t.result = probs
    return t


def demo(seed: int = 0) -> Trace:
    """Classify a small 6×6 image with vertical stripes into one of 2 classes."""
    rng = np.random.default_rng(seed)
    image = np.array(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        ]
    )
    kernel = np.array(
        [
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
        ]
    )
    # conv2d(6x6, 3x3) -> 4x4; max_pool2d(4x4, size=2) -> 2x2 -> flatten -> len 4.
    dense_weights = rng.normal(scale=0.5, size=(2, 4))
    dense_bias = np.zeros(2)
    return cnn_forward_trace(image, kernel, dense_weights, dense_bias, pool_size=2)

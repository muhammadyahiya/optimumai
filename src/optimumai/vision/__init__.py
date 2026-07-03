"""Computer vision fundamentals — how a CNN sees an image.

2D convolution (the local-pattern detector), pooling (translation-tolerant
downsampling), Sobel edge detection (a fixed convolution finding boundaries),
and a tiny CNN forward pass (conv -> ReLU -> pool -> flatten -> dense ->
softmax) that ties the others together and headlines the tensor-shape story.
"""

from optimumai.vision.cnn import cnn_forward, cnn_forward_trace, dense, relu
from optimumai.vision.convolution import conv2d, conv2d_trace
from optimumai.vision.edges import sobel_edges, sobel_edges_trace
from optimumai.vision.pooling import avg_pool2d, max_pool2d, pool2d_trace

__all__ = [
    "avg_pool2d",
    "cnn_forward",
    "cnn_forward_trace",
    "conv2d",
    "conv2d_trace",
    "dense",
    "max_pool2d",
    "pool2d_trace",
    "relu",
    "sobel_edges",
    "sobel_edges_trace",
]

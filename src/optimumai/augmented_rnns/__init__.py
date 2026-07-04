"""Attention and Augmented Recurrent Neural Networks.

Based on distill.pub's *Attention and Augmented Recurrent Neural Networks*
(2016): three RNN-era ideas that gave networks capabilities beyond a fixed
hidden state — attention as differentiable memory access, Neural Turing
Machines' external read/write memory, and Adaptive Computation Time's learned,
variable compute. All three converge on the same core trick (score, softmax,
weighted blend) that later became transformer attention; see
:mod:`optimumai.transformers.attention` for that descendant.
"""

from optimumai.augmented_rnns.act import (
    adaptive_computation_time,
    adaptive_computation_time_trace,
)
from optimumai.augmented_rnns.attention import attention_read, attention_read_trace
from optimumai.augmented_rnns.ntm import NTMMemory, ntm_read, ntm_trace, ntm_write

__all__ = [
    "NTMMemory",
    "adaptive_computation_time",
    "adaptive_computation_time_trace",
    "attention_read",
    "attention_read_trace",
    "ntm_read",
    "ntm_trace",
    "ntm_write",
]

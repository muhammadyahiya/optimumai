"""Frontier concepts — how today's large models are actually built and run.

FlashAttention (IO-aware tiling + online softmax), quantization (int8/int4),
LoRA (parameter-efficient fine-tuning), and DPO (preference alignment).
"""

from optimumai.frontier.flash_attention import flash_attention, flash_attention_trace
from optimumai.frontier.lora import lora, lora_trace
from optimumai.frontier.quantization import dequantize, quantize, quantize_trace
from optimumai.frontier.rlhf import dpo, dpo_trace

__all__ = [
    "dequantize",
    "dpo",
    "dpo_trace",
    "flash_attention",
    "flash_attention_trace",
    "lora",
    "lora_trace",
    "quantize",
    "quantize_trace",
]

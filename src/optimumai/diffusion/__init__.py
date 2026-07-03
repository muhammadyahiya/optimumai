"""Diffusion — forward noising and the reverse denoising idea."""

from optimumai.diffusion.schedule import forward_diffusion, forward_diffusion_trace

__all__ = ["forward_diffusion", "forward_diffusion_trace"]

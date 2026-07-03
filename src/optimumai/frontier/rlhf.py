"""RLHF and DPO — teaching a model *which* answers humans prefer.

A pretrained model predicts plausible text; alignment makes it predict *helpful,
harmless, honest* text. The classic recipe is RLHF:

1. **SFT** — supervised fine-tune on curated demonstrations.
2. **Reward model** — collect human preference pairs (chosen ≻ rejected) and
   train a model that scores a response.
3. **PPO** — reinforcement-learn the policy to maximize that reward while a KL
   penalty keeps it near the reference (SFT) model so it does not drift.

**DPO (Direct Preference Optimization)** proves that the RLHF optimum has a
closed form, letting you collapse steps 2–3 into a single classification-style
loss over the *same* preference pairs — no separate reward model, no RL loop:

    L = -log σ( β · [ (logπ_chosen − logπ_ref_chosen)
                      − (logπ_rejected − logπ_ref_rejected) ] )

The bracketed quantity is the difference of two *implicit rewards*
``r = β·(logπ − logπ_ref)``: DPO simply pushes the chosen reward above the
rejected one.
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def _sigmoid(z: float) -> float:
    """Numerically stable logistic sigmoid σ(z) = 1 / (1 + e^{-z})."""
    if z >= 0:
        return 1.0 / (1.0 + float(np.exp(-z)))
    ez = float(np.exp(z))
    return ez / (1.0 + ez)


def dpo_trace(prompt: str = "Explain gravity.", beta: float = 0.1, seed: int = 0) -> Trace:
    """Build the full trace of the DPO loss on one toy preference triple.

    Uses small seeded per-token log-probabilities for a policy model and a frozen
    reference model over a *chosen* and a *rejected* response, then walks through
    the implicit rewards, the preference margin, and the final DPO loss.
    """
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")

    rng = np.random.default_rng(seed)

    chosen = "Gravity is the attraction between masses."
    rejected = "Gravity is a kind of magic, who knows."

    # Toy per-token log-probs (always negative). The policy is nudged to like the
    # chosen response a little more and the rejected one a little less than ref.
    ref_chosen_tok = -rng.uniform(0.2, 1.0, size=6)
    ref_rejected_tok = -rng.uniform(0.2, 1.0, size=6)
    pi_chosen_tok = ref_chosen_tok + 0.15  # policy raised the chosen tokens
    pi_rejected_tok = ref_rejected_tok - 0.10  # policy lowered the rejected tokens

    # Sequence log-prob = sum of token log-probs.
    logpi_c = float(np.sum(pi_chosen_tok))
    logpi_r = float(np.sum(pi_rejected_tok))
    logref_c = float(np.sum(ref_chosen_tok))
    logref_r = float(np.sum(ref_rejected_tok))

    t = Trace(
        op="dpo",
        formula=(
            "L = -log σ( β·[ (logπ_chosen − logπ_ref_chosen) "
            "− (logπ_rejected − logπ_ref_rejected) ] )"
        ),
        complexity="O(1) per pair — one forward pass of policy + reference, no RL loop",
        why_ai=[
            "Preference pairs (chosen ≻ rejected) are how models learn "
            "helpfulness and harmlessness from human judgment",
            "DPO reparameterizes the RLHF optimum into a simple classification "
            "loss → more stable, no reward model, less reward hacking",
            "This is a simplified view of how assistants like Claude and GPT are "
            "aligned to human preferences",
        ],
        meta={"beta": beta, "prompt": prompt},
    )

    t.add(
        "The preference triple",
        f"prompt   = {prompt!r}\nchosen   = {chosen!r}\nrejected = {rejected!r}",
        detail="A human labeled 'chosen' as the better response for this prompt.",
    )

    t.add(
        "Sequence log-probs under the policy π",
        f"logπ(chosen)   = Σ tokens = {num(logpi_c)}\n"
        f"logπ(rejected) = Σ tokens = {num(logpi_r)}",
        detail=f"Per-token logπ(chosen) = {arr(pi_chosen_tok)}; a sequence "
        "log-prob is just the sum of its token log-probs.",
    )
    t.add(
        "Sequence log-probs under the frozen reference π_ref",
        f"logπ_ref(chosen)   = {num(logref_c)}\n"
        f"logπ_ref(rejected) = {num(logref_r)}",
        detail="π_ref is the SFT model; it anchors the policy so it cannot drift too far.",
    )

    r_chosen = beta * (logpi_c - logref_c)
    r_rejected = beta * (logpi_r - logref_r)
    t.add(
        "Implicit reward r = β·(logπ − logπ_ref)",
        f"r(chosen)   = {num(beta)}·({num(logpi_c)} − {num(logref_c)}) = {num(r_chosen)}\n"
        f"r(rejected) = {num(beta)}·({num(logpi_r)} − {num(logref_r)}) = {num(r_rejected)}",
        detail="DPO never trains a reward model — the reward is implied by how far "
        "the policy has moved from the reference.",
    )

    margin = r_chosen - r_rejected
    t.add(
        "Preference margin (chosen − rejected)",
        f"margin = r(chosen) − r(rejected) = {num(r_chosen)} − {num(r_rejected)} "
        f"= {num(margin)}",
        float(margin),
        detail="A positive margin means the policy already prefers the chosen "
        "response; the loss pushes it further positive.",
    )

    sig = _sigmoid(margin)
    loss = -float(np.log(sig))
    t.add(
        "DPO loss L = -log σ(margin)",
        f"σ({num(margin)}) = {num(sig)}   →   L = -log {num(sig)} = {num(loss)}",
        float(loss),
        detail="Minimizing L drives σ(margin) → 1, i.e. the chosen reward far "
        "above the rejected reward. L ≥ 0 always.",
    )

    t.add(
        "What DPO replaces in the RLHF pipeline",
        "RLHF = (1) SFT  →  (2) train a reward model on preference pairs  →  "
        "(3) PPO the policy against that reward + a KL penalty to π_ref.",
        detail="DPO collapses steps 2–3 into this one closed-form loss: no reward "
        "model, no reinforcement-learning loop, trained directly on the same pairs.",
    )

    t.result = float(loss)
    t.meta["reward_chosen"] = r_chosen
    t.meta["reward_rejected"] = r_rejected
    t.meta["margin"] = float(margin)
    return t


def dpo(
    prompt: str = "Explain gravity.",
    beta: float = 0.1,
    seed: int = 0,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.ENGINEER,
) -> float:
    """Return the DPO loss for one preference triple. ``explain=True`` prints the trace."""
    t = dpo_trace(prompt=prompt, beta=beta, seed=seed)
    return t.render(level) if explain else t.result


def demo() -> Trace:
    """Return a ready-to-render DPO trace with default settings."""
    return dpo_trace()

"""Learn the LLM fine-tuning pipeline on OptimumAI's own numpy/frontier primitives.

Pretraining teaches a model to predict plausible text; fine-tuning adapts that
pretrained model to a task or a behavior. This tutorial walks the whole modern
pipeline — SFT, LoRA, QLoRA, DPO, RLHF/PPO — with runnable numpy toys built on
`optimumai.frontier` and `optimumai.rl`, then shows the real production stack
(HuggingFace `transformers` + `peft` + `trl`) so you can read the actual API
even without those packages installed.

    from optimumai.tutorials import get_tutorial
    get_tutorial("finetuning").run()
"""

from __future__ import annotations

from optimumai.tutorials.core import Tutorial


def build() -> Tutorial:
    t = Tutorial(
        name="finetuning",
        title="LLM fine-tuning, from SFT to LoRA to DPO",
        summary="The fine-tuning pipeline on OptimumAI's numpy primitives, then real HF/PEFT/TRL.",
    )

    # --------------------------------------------------------------- what is FT
    t.md(
        "## 1. What is fine-tuning?\n\n"
        "**Pretraining** trains a model from scratch on a huge unlabeled corpus "
        "to predict the next token — it learns grammar, facts, and reasoning "
        "patterns, but not *how to behave*. **Fine-tuning** takes those pretrained "
        "weights and continues training on a smaller, curated dataset to adapt "
        "them to a task or a behavior:\n\n"
        "- **SFT** (supervised fine-tuning) — train on (prompt, ideal response) "
        "pairs so the model imitates a demonstrated behavior (e.g. 'be a helpful "
        "assistant', not just 'complete this text').\n"
        "- **Alignment** — further tune the SFT model using *human preferences* "
        "(chosen vs. rejected responses) via RLHF (reward model + PPO) or the "
        "more direct DPO loss, so it prefers helpful/harmless answers.\n\n"
        "Every stage below reuses the *same* weight-update machinery — the only "
        "thing that changes is what the loss is computed over."
    )

    # ----------------------------------------------------------------------- SFT
    t.md(
        "## 2. SFT — supervised fine-tuning\n\n"
        "Concretely, SFT is: predict, compute a loss against the target label, "
        "backprop, step — exactly the training loop from the PyTorch tutorial, "
        "just pointed at (prompt, response) pairs instead of arbitrary regression "
        "targets. Below, a toy classifier stands in for 'pick the right token': "
        "we fine-tune a small `optimumai.neural_networks.MLP` to map 4 toy "
        "'prompt embeddings' to their labeled 'response class' and watch the loss "
        "fall."
    )
    t.code(
        """
from optimumai.neural_networks import MLP
from optimumai.optimization import SGD

# Toy "prompts" (already embedded as vectors) -> target "response" scores.
prompts = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, -1.0]]
targets = [1.0, -1.0, 0.5, -0.5]

model = MLP(n_in=2, n_outs=[4, 1], seed=1)
opt = SGD(model.parameters(), lr=0.05)

losses = []
for epoch in range(60):
    preds = [model(p) for p in prompts]
    loss = sum((p - y) ** 2 for p, y in zip(preds, targets, strict=True)) * (1.0 / len(prompts))
    for param in model.parameters():
        param.grad = 0.0
    loss.backward()
    opt.step()
    losses.append(loss.data)

print(f"SFT loss[0]={losses[0]:.4f} -> loss[-1]={losses[-1]:.4f}")
print("loss decreased:", losses[-1] < losses[0])
""",
        note="optimumai: a toy SFT loop — imitate labeled (prompt, response) pairs",
    )

    # ---------------------------------------------------------------------- LoRA
    t.md(
        "## 3. LoRA — Low-Rank Adaptation\n\n"
        "Full fine-tuning updates *every* weight in `W0` (`d_in * d_out` numbers). "
        "LoRA freezes `W0` and learns a low-rank update `W = W0 + B @ A`, where "
        "`B` is `d_out x r`, `A` is `r x d_in`, and `r` is tiny. `B` starts at "
        "zero and `A` starts as noise, so `B @ A = 0` at init — training begins "
        "from exactly the pretrained model. `optimumai.frontier.lora.lora_trace` "
        "builds this whole picture; we call it here and print the parameter "
        "savings it computes."
    )
    t.code(
        """
from optimumai.frontier.lora import lora_trace

trace = lora_trace(d_in=64, d_out=64, rank=4, seed=0)
print(f"full params={trace.meta['full_params']}, lora params={trace.meta['lora_params']}")
print(f"reduction factor={trace.meta['reduction_factor']:.2f}x fewer trainable params")
""",
        note="optimumai: W = W0 + B@A, B starts at zero -> ~exact param-count saving",
    )

    # --------------------------------------------------------------------- QLoRA
    t.md(
        "## 4. QLoRA — quantized base + LoRA adapters\n\n"
        "QLoRA's idea: keep the (huge, frozen) base weights in 4-bit precision to "
        "cut memory, and train only small full-precision LoRA adapters on top. "
        "The base model is quantized once with `optimumai.frontier.quantization` "
        "(`q = round(x/scale) + zero_point`, dequantized on the fly for compute); "
        "the LoRA `A`/`B` factors from above stay in fp32 and carry all the "
        "gradients. Below we quantize a toy 'base weight' matrix to int4 and "
        "confirm the memory saving and the reconstruction error."
    )
    t.code(
        """
import numpy as np

from optimumai.frontier.quantization import quantize_trace

rng = np.random.default_rng(0)
W0 = rng.standard_normal((64, 64))  # the frozen "pretrained" base weight

trace = quantize_trace(W0, bits=4, scheme="symmetric")
print(f"compression={trace.meta['compression_ratio']:.1f}x smaller than fp32")
print(f"max reconstruction error={trace.meta['max_error']:.4f}")
print("QLoRA = this int4 frozen base + fp32 LoRA A/B adapters trained on top")
""",
        note="optimumai: int4-quantize the frozen base; LoRA adapters stay full precision",
    )

    # ------------------------------------------------------------ preference: DPO
    t.md(
        "## 5. Preference alignment — DPO\n\n"
        "After SFT, we want the model to prefer *better* responses, not just "
        "imitate one demonstration. Given a (chosen, rejected) pair for the same "
        "prompt, DPO defines an implicit reward `r = beta * (logpi - logpi_ref)` "
        "for each response and minimizes "
        "`L = -log sigmoid(r_chosen - r_rejected)` — pure supervised "
        "classification, no reward model, no RL rollout. "
        "`optimumai.frontier.rlhf.dpo_trace` builds exactly this on a toy "
        "preference triple."
    )
    t.code(
        """
from optimumai.frontier.rlhf import dpo_trace

trace = dpo_trace(prompt="Explain gravity.", beta=0.1, seed=0)
print(f"DPO loss={trace.result:.4f}")
print(f"reward margin (chosen - rejected)={trace.meta['margin']:.4f}")
""",
        note="optimumai: DPO loss on one (chosen, rejected) preference pair",
    )
    t.md(
        "### RLHF / PPO — the road DPO paved over\n\n"
        "Classic RLHF does not compute that closed-form loss directly. Instead: "
        "(1) SFT, (2) train a separate **reward model** on preference pairs, "
        "(3) **PPO** the policy to maximize that reward while a KL penalty keeps "
        "it near the SFT reference model. PPO's clipped objective "
        "(`optimumai.rl.ppo.ppo_clip_trace`) is what actually updates the policy "
        "in step (3) — token by token, on-policy, with rollouts. **DPO vs PPO**: "
        "PPO explores beyond the labeled pairs via sampling but needs a reward "
        "model, rollouts, and careful clipping to stay stable; DPO trades that "
        "machinery for a single supervised loss directly on the preference data, "
        "at the cost of never generating anything the labeled pairs didn't cover."
    )
    t.code(
        """
from optimumai.rl.ppo import ppo_clip_trace

trace = ppo_clip_trace()  # a hand-built batch exercising every clip case
print(f"PPO clipped loss={trace.result:.4f}")
print(f"samples where clipping engaged={trace.meta['n_clipped']}/{trace.meta['batch_size']}")
""",
        note="optimumai: PPO's clipped surrogate objective — the RL half of RLHF",
    )

    # ------------------------------------------------------------- production stack
    t.md(
        "## 6. The production stack: transformers + peft + trl\n\n"
        "In practice you do not write the training loop by hand — HuggingFace's "
        "stack wires it up for you:\n\n"
        "- **`transformers`** loads the pretrained model/tokenizer.\n"
        "- **`peft`** wraps it with LoRA adapters (`LoraConfig` + `get_peft_model`).\n"
        "- **`trl`** provides trainer classes for SFT (`SFTTrainer`) and preference "
        "alignment (`DPOTrainer`).\n\n"
        "Key hyperparameters: learning rate (SFT ~1e-5-2e-5 full fine-tune, "
        "~1e-4-3e-4 for LoRA since fewer params move faster), LoRA rank `r` "
        "(4-64: higher r = more capacity, more params, more overfit risk), "
        "`lora_alpha` (scales the update, commonly `2*r`), and epochs (SFT "
        "usually needs only 1-3 passes over instruction data). Watch for "
        "**overfitting** on small fine-tuning sets and **catastrophic "
        "forgetting** (the model loses pretrained capabilities) — both are why "
        "LoRA's small, easily-discarded adapters and DPO's KL-anchored loss "
        "against a reference model are popular default choices."
    )
    t.code(
        """
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
""",
        note="the real HuggingFace transformers — load a pretrained base model",
        requires=("transformers",),
    )
    t.code(
        """
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,               # rank
    lora_alpha=16,     # scales the update (commonly 2*r)
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
""",
        note="the real PEFT — wrap the base model with LoRA adapters",
        requires=("peft",),
    )
    t.code(
        """
from trl import SFTConfig, SFTTrainer

sft_config = SFTConfig(output_dir="./sft-out", num_train_epochs=2, learning_rate=2e-4)
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,  # a dataset of {"text": "<prompt+response>"}
)
trainer.train()
""",
        note="the real TRL — supervised fine-tuning with LoRA adapters attached",
        requires=("trl",),
    )
    t.code(
        """
from trl import DPOConfig, DPOTrainer

dpo_config = DPOConfig(output_dir="./dpo-out", beta=0.1, learning_rate=5e-5)
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,  # None reuses the base model's frozen weights as reference
    args=dpo_config,
    train_dataset=preference_dataset,  # {"prompt", "chosen", "rejected"}
)
dpo_trainer.train()
""",
        note="the real TRL — preference alignment with DPO, no reward model needed",
        requires=("trl",),
    )

    t.md(
        "## Where this lives in OptimumAI\n\n"
        "`optimumai.frontier.lora`, `optimumai.frontier.quantization`, and "
        "`optimumai.frontier.rlhf` (DPO) each expose a `*_trace()` function you "
        "can render with `explain=True`-style output for a fully narrated, "
        "step-by-step walkthrough; `optimumai.rl.ppo.ppo_clip_trace` does the "
        "same for PPO's clipped objective."
    )

    return t

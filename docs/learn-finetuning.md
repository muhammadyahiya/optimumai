# Learn LLM Fine-Tuning

Fine-tuning is how a model that merely predicts plausible text becomes a model
that behaves the way you want. This page walks the whole modern pipeline —
what fine-tuning is, why full fine-tuning is expensive, how LoRA and QLoRA
make it cheap, and how DPO and RLHF/PPO align a model to human preferences —
with runnable OptimumAI code proving every mechanism at toy scale, and labeled
HuggingFace reference code showing the real production stack.

!!! tip "Run this interactively"
    ```bash
    optimumai tutorial finetuning
    ```

    or from Python:

    ```python
    from optimumai.tutorials import get_tutorial
    get_tutorial("finetuning").run()
    ```

    That tutorial is a runnable, numpy-only toy pipeline — SFT → LoRA → QLoRA →
    DPO → PPO — built entirely on `optimumai.frontier` and `optimumai.rl`, plus
    the same HuggingFace reference code shown on this page (labeled, not
    executed, since `transformers`/`peft`/`trl` are not installed here either).

!!! warning "Labeling convention on this page"
    Blocks marked **OptimumAI (runnable)** were executed against this repo's
    `optimumai.frontier`/`optimumai.rl` modules while writing this page — the
    exact numbers shown are real output. Blocks marked **HuggingFace
    (reference, not run here)** are correct, idiomatic `transformers`/`peft`/
    `trl` usage, not executed in this environment.

---

## 1. What is fine-tuning?

**Intuition.** **Pretraining** trains a model from scratch on a huge corpus of
unlabeled text to do one thing: predict the next token. Do that at enough
scale and the model absorbs grammar, facts, reasoning patterns, and style —
but nothing in that objective teaches it *how to behave* when a person is
talking to it. A pretrained base model, asked a question, is just as likely to
continue it with more questions, or hallucinate a forum thread around it, as
to answer it.

**Fine-tuning** takes those pretrained weights and continues training on a
smaller, curated dataset to adapt the model toward a task or a behavior.
Modern LLM fine-tuning is usually a pipeline of stages, each building on the
last:

```
pretrain  →  SFT  →  preference alignment (DPO  or  RLHF/PPO)
(next-token   (imitate demonstrated    (prefer human-judged
 prediction    good behavior)           better responses)
 on raw text)
```

- **Pretrain** — next-token prediction on raw text at massive scale. Produces
  a "base model": fluent, knowledgeable, but not obedient or safe by default.
- **SFT (supervised fine-tuning)** — train on (prompt, ideal response) pairs
  so the model imitates a demonstrated behavior: "be a helpful assistant,"
  not just "complete this text plausibly."
- **Preference alignment** — further tune the SFT model using *human
  preferences* between pairs of responses (chosen vs. rejected), via either
  classic **RLHF** (train a reward model, then reinforcement-learn against it
  with **PPO**) or the more direct **DPO** loss, which skips the reward model
  and the RL loop entirely.

Every stage reuses the *exact same* weight-update machinery you already know
from [Learn PyTorch](learn-pytorch.md) — predict, compute a loss, backward,
step. The only thing that changes stage to stage is **what the loss is
computed over**: raw text likelihood (pretrain), demonstrated-response
likelihood (SFT), or a function of a *pair* of responses (DPO/RLHF).

---

## 2. SFT — supervised fine-tuning

**Intuition.** SFT is ordinary supervised learning: you have (prompt, ideal
response) pairs, and you train the model to assign higher probability to the
ideal response given the prompt — the same next-token-prediction loss as
pretraining, just now computed only over curated, high-quality
demonstrations instead of the open internet.

A toy stand-in makes the mechanics concrete: fine-tune a small
`optimumai.neural_networks.MLP` to map "prompt embeddings" (stand-ins for
whatever a real tokenizer + embedding table would produce) to a target
"response" value, and watch the loss fall — the identical predict → loss →
backward → step loop from the PyTorch training-loop section, just pointed at
(prompt, response) data.

```python
# OptimumAI (runnable)
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
```

Real SFT differs only in scale and in what the loss is computed over: instead
of a scalar regression target, it's cross-entropy over vocabulary-sized
logits, summed over every generated token in the response, with the prompt
tokens typically masked out of the loss (you want the model to learn to
*generate* the response, not to re-predict the prompt it was given).

---

## 3. Full fine-tuning vs. parameter-efficient fine-tuning (PEFT)

**Intuition.** "Full fine-tuning" means every single weight in the model is
allowed to move — you back-propagate through the whole network and update
every parameter with the optimizer. That's expensive in a way that compounds:

- **Gradient memory** — you need a gradient the same size as every parameter.
- **Optimizer state** — Adam alone stores two extra moment buffers per
  parameter (first and second moment), so Adam's memory overhead is
  *2x the model size*, on top of the weights and gradients themselves.
- **Checkpoint size** — a full fine-tuned checkpoint is a full copy of the
  model. Want five fine-tuned variants of a 7B model? That's five full 7B
  checkpoints on disk.

**Parameter-efficient fine-tuning (PEFT)** is the family of techniques that
freeze almost all of the pretrained weights and train only a small number of
*new* parameters layered on top. You get most of the behavioral change at a
tiny fraction of the memory, compute, and storage cost — and because the base
model is untouched, the same frozen base can serve many different fine-tuned
"adapters" simultaneously. **LoRA** is by far the most widely used PEFT
method, and is the one this page (and OptimumAI's `frontier` package) focuses
on.

---

## 4. LoRA — `W = W₀ + BA`

**Intuition.** LoRA (Low-Rank Adaptation) starts from an empirical
observation: the *update* a model's weights need during fine-tuning tends to
have low "intrinsic rank" — it lives in a much smaller subspace than the full
weight matrix's dimensions would suggest. So instead of learning a full
`d_out × d_in` update matrix `ΔW`, LoRA factors it into two small matrices,
`B` (`d_out × r`) and `A` (`r × d_in`), with rank `r ≪ min(d_in, d_out)`, and
learns only those:

```
W = W₀ + ΔW = W₀ + B·A
```

`W₀` (the pretrained weight) is **frozen** — no gradient ever touches it.
Only `A` and `B` are trainable. The trick that makes this safe to drop into a
pretrained model with zero risk of breaking it: **`B` is initialized to all
zeros**, and `A` to random Gaussian noise. Since `ΔW = B·A` and `B` starts at
zero, `ΔW = 0` at the very first training step — the adapted model is
*mathematically identical* to the pretrained model until training actually
moves `B` off zero.

### Why the parameter count savings are so large

A full fine-tune of one `d_in × d_out` weight matrix has `d_in · d_out`
trainable parameters. LoRA's `A` and `B` together have `r · (d_in + d_out)`
— linear in `d_in + d_out` rather than quadratic in their product. The
reduction factor is:

```
full_params / lora_params = (d_in · d_out) / (r · (d_in + d_out))
```

This is exactly what `optimumai.frontier.lora.lora_trace` computes and
narrates, step by step — freezing a seeded `W₀`, initializing `A` (Gaussian)
and `B` (zeros), forming `ΔW = B·A`, and counting parameters:

```python
# OptimumAI (runnable)
from optimumai.frontier.lora import lora_trace

trace = lora_trace(d_in=64, d_out=64, rank=4, seed=0)
print(f"full params={trace.meta['full_params']}, lora params={trace.meta['lora_params']}")
print(f"reduction factor={trace.meta['reduction_factor']:.2f}x fewer trainable params")
# full params=4096, lora params=512, reduction factor=8.00x fewer trainable params
```

At a more realistic transformer-scale matrix (a 768×768 attention projection,
the size used by GPT-2-small/BERT-base), rank 8 gives a much larger win:

```python
# OptimumAI (runnable)
from optimumai.frontier.lora import lora

reduction = lora(d_in=768, d_out=768, rank=8)
print(f"{reduction:.0f}x fewer trainable params")   # 48x fewer trainable params
```

And on GPT-3 175B, the original LoRA paper reports roughly **10,000x** fewer
trainable parameters and about 3x less GPU memory than full fine-tuning —
the ratio grows with model size because `d_in · d_out` grows quadratically
while `r · (d_in + d_out)` stays linear.

`lora_trace` also demonstrates the "starts as a no-op" property directly: at
initialization, `(W₀ + B·A)·x` equals `W₀·x` exactly (max difference ~0,
limited only by floating-point rounding), and only after `B` is nudged off
zero (simulating a training step) does the output actually shift:

```python
# OptimumAI (runnable, continuing the trace above)
# At init:      max |y - W0@x|        ≈ 0            (B=0  ->  ΔW=0)
# After a step: max |y_trained - W0@x| > 0            (B≠0  ->  ΔW≠0, adapter is "doing work")
```

### Where LoRA is applied in a real transformer

In practice, LoRA is applied to specific weight matrices inside each
transformer block — most commonly the attention projections (`q_proj`,
`v_proj`, or in GPT-2's fused form, `c_attn`) — not to every weight in the
model. Applying it to more matrices (attention *and* MLP projections)
increases capacity and trainable-parameter count, trading some of LoRA's
efficiency for closer-to-full-fine-tune expressiveness.

---

## 5. QLoRA — 4-bit base + LoRA adapters

**Intuition.** LoRA already shrinks *trainable* parameters and optimizer
state dramatically, but the frozen base model still has to sit in memory in
full precision (typically fp16/bf16, 2 bytes per parameter) just to run the
forward pass. For a 70B-parameter model that's still ~140GB just for weights.
**QLoRA**'s contribution: quantize the frozen base weights down to **4-bit
integers**, and train LoRA adapters (kept in full precision) on top of that
quantized base. The base model shrinks ~4x in memory; only the tiny adapters
carry gradients and need optimizer state.

### The quantization math

`optimumai.frontier.quantization` implements exactly the scheme QLoRA (and
weight-only LLM quantization methods generally — LLM.int8(), GPTQ, AWQ) rely
on. Every weight is mapped onto a small integer grid using two numbers per
group of weights:

```
q  = clip(round(x / scale) + zero_point, qmin, qmax)     # quantize
x̂  = (q - zero_point) · scale                             # dequantize
```

Think of `scale` and `zero_point` like the *std* and *mean* of a
normalization: `scale` stretches the integer grid to cover the data's range,
and `zero_point` shifts the grid so real `0.0` lands exactly on an integer
code. **Symmetric** quantization (the common choice for weights, which are
roughly zero-mean) fixes `zero_point = 0` and sets `scale = max|x| / qmax`;
**asymmetric** quantization fits both ends of the range and is better suited
to skewed data like post-ReLU activations.

```python
# OptimumAI (runnable)
import numpy as np
from optimumai.frontier.quantization import quantize_trace

rng = np.random.default_rng(0)
W0 = rng.standard_normal((64, 64))  # the frozen "pretrained" base weight

trace = quantize_trace(W0, bits=4, scheme="symmetric")
print(f"compression={trace.meta['compression_ratio']:.1f}x smaller than fp32")
print(f"max reconstruction error={trace.meta['max_error']:.4f}")
# compression=8.0x smaller than fp32
# max reconstruction error=0.2785
```

(The `8.0x` here is int4 vs. fp32 specifically — 32 bits down to 4 is an 8x
reduction; QLoRA papers typically compare int4 against fp16/bf16, which is a
4x reduction, since fp16 is the usual training/inference precision to begin
with.)

That reconstruction error — the gap between the original float and the
value rebuilt from its quantized code — is the real cost of quantization: a
small amount of numerical noise on every weight, in exchange for a much
smaller memory footprint. **Per-channel** quantization (one `scale` per row of
a weight matrix, rather than one for the whole tensor) is the standard
mitigation: it stops one unusually large-magnitude row from blowing up the
`scale` — and therefore the rounding error — for every other row.

### Putting it together

QLoRA's recipe, end to end:

1. Load the pretrained base model, quantize its weights to 4-bit (using a
   scheme like NF4 — a 4-bit format tuned for the roughly-normal distribution
   of neural network weights — in the original QLoRA paper).
2. Attach LoRA adapters (`A`, `B`) in full precision (fp16/bf16) on top of
   the quantized base, exactly as in the LoRA section above.
3. Train only `A` and `B`. Every forward pass dequantizes the relevant base
   weights on the fly to compute in higher precision, then the low-rank
   update is added on top — but only `A`/`B` accumulate gradients or
   optimizer state.

This is why QLoRA made fine-tuning genuinely large open models feasible on a
single consumer or single-datacenter GPU: the expensive part (storing the
base model) shrinks ~4x, and the trainable part (LoRA adapters) was already
tiny.

---

## 6. Preference alignment: DPO vs. RLHF/PPO

Once you have an SFT model that can *follow* instructions, the next question
is which of several plausible responses is actually *better*. That's a
comparison humans are much better at judging than describing with a simple
loss function — so both approaches below start from the same kind of data:
**preference pairs** — a prompt, plus a `chosen` response and a `rejected`
response, labeled by a human (or another model acting as a judge).

### DPO — Direct Preference Optimization

**Intuition.** RLHF's reward-model-plus-RL machinery exists to solve one
problem: turning "humans prefer response A over response B" into a training
signal. DPO's insight is that, under reasonable assumptions, the *optimal*
policy from that whole RLHF procedure has a **closed form** — you can
write down the loss you'd end up optimizing toward directly, and just
optimize *that*, with a single supervised loss over the same preference
pairs. No reward model to train, no reinforcement-learning rollouts, no PPO
clipping to tune.

The DPO loss:

```
L = -log σ( β · [ (logπ(chosen) − logπ_ref(chosen))
                 − (logπ(rejected) − logπ_ref(rejected)) ] )
```

Read the bracketed term as a difference of two **implicit rewards**,
`r = β · (logπ − logπ_ref)` — "how much more likely does the *current* policy
make this response, relative to the frozen reference (SFT) model." DPO simply
pushes the chosen response's implicit reward above the rejected response's.
`β` controls how tightly the policy is anchored to the reference model — a
higher `β` permits less drift.

`optimumai.frontier.rlhf.dpo_trace` builds exactly this computation on one toy
preference triple (seeded per-token log-probabilities, nudged so the policy
already prefers the chosen response a little more than the reference does):

```python
# OptimumAI (runnable)
from optimumai.frontier.rlhf import dpo_trace

trace = dpo_trace(prompt="Explain gravity.", beta=0.1, seed=0)
print(f"DPO loss={trace.result:.4f}")
print(f"reward margin (chosen - rejected)={trace.meta['margin']:.4f}")
# DPO loss=0.6210
# reward margin (chosen - rejected)=0.1500
```

A positive margin means the policy already prefers the chosen response over
the rejected one; minimizing `L` drives that margin further positive (`σ` of
a large positive number approaches 1, and `-log(1) = 0`).

### RLHF / PPO — the road DPO paved over

Classic RLHF does not compute a closed-form loss directly. It's a three-stage
pipeline:

1. **SFT** — as above.
2. **Reward model** — collect a large set of human preference pairs and train
   a separate model whose job is just to output a scalar score for "how good
   is this response."
3. **PPO** — reinforcement-learn the policy (the language model itself,
   viewed as an agent that "acts" by choosing tokens) to maximize the reward
   model's score, while a KL-divergence penalty keeps the policy from
   drifting too far from the SFT reference model (which would otherwise let
   the policy "reward hack" — find degenerate outputs the reward model
   over-scores).

**PPO (Proximal Policy Optimization)** is what actually updates the policy in
step 3. Its core problem: a policy-gradient update is only strictly valid for
data sampled from the *current* policy, but collecting fresh rollouts for
every single gradient step is expensive — you want to reuse a batch of
rollouts for several optimizer steps. PPO makes that safe with a **clipped,
importance-weighted objective**:

```
r_t(θ) = exp( logπ_θ(a|s) − logπ_θ_old(a|s) )       # how much the policy has moved

L^CLIP(θ) = E[ min( r_t·A_t,  clip(r_t, 1−ε, 1+ε)·A_t ) ]
```

`A_t` is the **advantage** — how much better an action was than the state's
average outcome (positive: do more of this; negative: do less). Reading the
`min` of the two terms case by case: if the ratio has already moved past
`1±ε` in the direction the advantage favors, clipping caps the objective —
there's no extra reward for pushing the policy *even further* in one update.
Inside the trust region `r_t ∈ [1−ε, 1+ε]`, clipping never engages and
`L^CLIP` is just the plain surrogate `r_t·A_t`. The `min` (not the clipped
term alone) makes this a **pessimistic** bound: clipping only ever removes an
incentive to move further, never creates a new incentive to move backward.
Net effect: you can safely take several optimizer steps, even multiple
epochs, over one batch of rollouts.

`optimumai.rl.ppo.ppo_clip_trace` walks a hand-built batch that exercises
every clipping case (inside the trust region, clipped upside, clipped
downside, and the "ratio clamped but the min keeps the unclipped term
anyway" case):

```python
# OptimumAI (runnable)
from optimumai.rl.ppo import ppo_clip_trace

trace = ppo_clip_trace()   # a hand-built batch exercising every clip case
print(f"PPO clipped loss={trace.result:.4f}")
print(f"samples where clipping engaged={trace.meta['n_clipped']}/{trace.meta['batch_size']}")
# PPO clipped loss=0.1774
# samples where clipping engaged=2/6
```

### DPO vs. PPO — the trade-off

| | **DPO** | **RLHF / PPO** |
| --- | --- | --- |
| Reward model | None — reward is implicit in the policy/reference log-prob gap | A separate model, trained on preference pairs |
| Training loop | One supervised classification loss | Reward model training + an on-policy RL loop (rollouts, advantages, clipping) |
| Stability | Generally more stable, fewer moving parts | Sensitive to reward-model quality, KL coefficient, rollout hyperparameters |
| Exploration | Limited to the labeled pairs — cannot generate and rate novel responses during training | Can explore beyond the labeled data via sampling, at the cost of reward-hacking risk |
| Engineering cost | Lower — no rollout infrastructure needed | Higher — needs a reward model and an RL training loop |

In practice, DPO has become the default first choice for preference alignment
because it's dramatically simpler to implement and tune correctly, while full
RLHF/PPO remains relevant when you specifically need the policy to explore
and be scored on responses it generates during training, not just the
responses present in a fixed preference dataset.

---

## 7. The production stack: `transformers` + `peft` + `trl`

In practice, you rarely hand-write the training loop for any of the stages
above — HuggingFace's ecosystem wires it up for you. This is real, correct
usage; it is **not executed** in this docs environment.

```python
# HuggingFace (reference, not run here)
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

```python
# HuggingFace (reference, not run here) — attach LoRA adapters with peft
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                          # rank
    lora_alpha=16,                # scales the update (commonly 2*r)
    target_modules=["c_attn"],    # which weight matrices get an adapter
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # confirms the tiny fraction that's trainable
```

```python
# HuggingFace (reference, not run here) — SFT with trl
from trl import SFTConfig, SFTTrainer

sft_config = SFTConfig(output_dir="./sft-out", num_train_epochs=2, learning_rate=2e-4)
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,   # a dataset of {"text": "<prompt+response>"}
)
trainer.train()
```

```python
# HuggingFace (reference, not run here) — preference alignment with DPO, no reward model
from trl import DPOConfig, DPOTrainer

dpo_config = DPOConfig(output_dir="./dpo-out", beta=0.1, learning_rate=5e-5)
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,   # None reuses the base model's frozen weights as the reference
    args=dpo_config,
    train_dataset=preference_dataset,   # {"prompt", "chosen", "rejected"}
)
dpo_trainer.train()
```

For classic RLHF/PPO, `trl` also ships a `PPOTrainer` that coordinates
generating rollouts from the current policy, scoring them with a reward
model, and running PPO updates — considerably more moving parts than
`SFTTrainer`/`DPOTrainer`, which is exactly the added engineering cost the
DPO-vs-PPO table above is describing.

For QLoRA specifically, the `bitsandbytes` library provides the 4-bit
quantized model loading (`load_in_4bit=True` in
`AutoModelForCausalLM.from_pretrained(...)`, with `BitsAndBytesConfig`
controlling the quantization scheme), combined with the same `peft`
`LoraConfig`/`get_peft_model` calls shown above.

---

## 8. Hyperparameters

| Hyperparameter | Typical range | Notes |
| --- | --- | --- |
| Learning rate (full fine-tune) | ~1e-5 to 2e-5 | Small — you're nudging an already-good model, not training from scratch. |
| Learning rate (LoRA) | ~1e-4 to 3e-4 | Higher than full fine-tuning is typical: far fewer parameters move, so each can safely take bigger steps. |
| LoRA rank `r` | 4 to 64 | Higher `r` = more capacity (closer to full fine-tuning) but more trainable params and more overfitting risk on small datasets. |
| LoRA `alpha` | commonly `2 × r` | Scales the LoRA update (`ΔW` is scaled by `alpha / r`); higher `alpha` relative to `r` makes the adapter's contribution stronger. |
| Epochs (SFT) | 1-3 | Instruction-tuning datasets are usually small relative to pretraining; more than a few passes tends to overfit or memorize. |
| Epochs (DPO) | 1-3 | Similarly small — preference datasets are typically far smaller than SFT datasets. |
| Batch size | as large as memory allows, often with gradient accumulation | Larger batches give a less noisy gradient estimate; gradient accumulation simulates a larger batch on limited memory by summing gradients over several forward/backward passes before stepping. |
| DPO `β` | ~0.1 (commonly 0.01-0.5) | Controls how tightly the policy is anchored to the reference model — higher `β` means less allowed drift per unit of preference margin. |
| PPO `ε` (clip range) | ~0.1-0.2 | How far the probability ratio is allowed to move before clipping engages. |

---

## 9. Pitfalls

- **Overfitting.** Fine-tuning datasets are tiny compared to pretraining
  corpora — a model can memorize a few thousand examples in a couple of
  epochs. Watch validation loss, not just training loss, and prefer more
  data over more epochs when you can get it.
- **Catastrophic forgetting.** Aggressively fine-tuning on a narrow
  distribution of data can degrade capabilities the base model had before —
  the model "forgets" general knowledge or skills while specializing.
  LoRA's small, easily-discarded adapters (you can always fall back to the
  untouched base) and DPO's KL-anchored loss against a frozen reference model
  are both popular defaults partly *because* they limit how far the model is
  allowed to drift, which mitigates this.
- **Data quality over data quantity.** A smaller set of carefully curated,
  correct (prompt, response) or (chosen, rejected) pairs consistently
  outperforms a larger set of noisy ones — this is one of the most
  reproducible findings across SFT and preference-alignment work. Garbage
  preference labels teach the model a garbage notion of "better."
  Comparably-styled surface-level differences (chosen response is just
  longer, or more formal) can also get learned as spurious shortcuts instead
  of the actual quality difference you intended.
- **Reward hacking (RLHF/PPO specifically).** If the reward model is
  imperfect (it always is), the policy can find outputs that score highly on
  the reward model without actually being good — e.g. degenerate repetition,
  or exploiting a length bias in the reward model. The KL penalty to the
  reference model, and reward model quality/coverage, are the main levers
  against this. DPO sidesteps this specific failure mode by never training
  or optimizing against a separate learned reward model in the first place.
- **Evaluation.** Loss curves going down does not guarantee the model got
  *better* at the thing you care about — always pair fine-tuning with a held-
  out evaluation that reflects the actual task (task-specific accuracy, a
  preference win-rate against the previous checkpoint, or a rubric-based
  judgment), not just perplexity on held-out fine-tuning data. See
  `optimumai.evaluation` (`optimumai eval bleu`, `optimumai eval
  faithfulness`, and friends) for OptimumAI's own explainable metrics if you
  want a fast, dependency-free starting point for this.

---

## Where this lives in OptimumAI

Every mechanism on this page has a runnable, narrated counterpart:

```bash
optimumai learn lora           # low-rank adapters — parameter reduction, step by step
optimumai learn dpo            # the DPO loss on one preference triple
optimumai quantize "[0.1,-2.3,4.5,3.14]" --bits 4    # quantize YOUR numbers, see the error
optimumai rl ppo                # the PPO clipped objective
```

```python
from optimumai.frontier import lora, quantize, dpo
from optimumai.rl import ppo_clip

lora(d_in=768, d_out=768, rank=8, explain=True)      # narrated LoRA trace
dpo(prompt="Explain gravity.", beta=0.1, explain=True)  # narrated DPO trace
ppo_clip(explain=True)                                 # narrated PPO trace
```

`optimumai.frontier.lora`, `optimumai.frontier.quantization`, and
`optimumai.frontier.rlhf` (DPO) each expose a `*_trace()` function you can
render for a fully narrated, step-by-step walkthrough at your chosen
`level=` (`beginner` → `researcher`); `optimumai.rl.ppo.ppo_clip_trace` does
the same for PPO's clipped objective. Run `optimumai tutorial finetuning` for
the complete, runnable toy pipeline that chains all of these together — SFT,
LoRA, QLoRA, DPO, and PPO — end to end.

!!! note "Where to start"
    If any of the mechanics above felt unfamiliar, back up: [Learn NumPy](learn-numpy.md)
    covers the array operations everything here is built from, and
    [Learn PyTorch](learn-pytorch.md) covers the `nn.Module`/autograd/optimizer
    machinery that a real fine-tuning run sits on top of.

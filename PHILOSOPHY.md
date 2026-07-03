# The OptimumAI Philosophy

> *Unlock the math behind AI.* Not by describing it — by **running** it, one
> transparent step at a time.

OptimumAI is built on a simple conviction: **you understand a system when you can
rebuild its atoms.** Modern AI looks like magic only because its fundamentals are
hidden behind billions of parameters and highly optimized kernels. Strip those
away and what remains is small, elegant, and learnable — a dot product, a chain
rule, an energy. This library makes those atoms visible.

Three thinkers shape how we build.

---

## 1. Karpathy — build it from scratch, spelled out

Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) is ~150
lines that contain the whole of backpropagation; [nanoGPT](https://github.com/karpathy/nanoGPT)
is a GPT you can read in an afternoon. The lesson is that **the fundamentals are
tiny** — a scalar `Value` with a `_backward` closure is enough to train a neural
network. His [recipe for training neural nets](http://karpathy.github.io/2019/04/25/recipe/)
adds the discipline: *"become one with the data,"* visualize everything, overfit
a small model before you regularize.

**How OptimumAI applies it**
- `autograd/value.py` *is* micrograd, plus a `backward_trace()` that shows the
  chain rule flowing backward through the graph.
- `neural_networks/` builds `Neuron → Layer → MLP` on that one `Value` type.
  There is no separate "neural network math" — just the autograd engine composed
  a few thousand times.
- Every operation is a few dozen readable lines. Small is a feature.

## 2. LeCun — understanding is a predictive world model

Yann LeCun argues that intelligence is grounded in **world models**: systems that
predict *what happens next* in an abstract representation space rather than at the
level of raw pixels. His [JEPA](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/)
(Joint-Embedding Predictive Architecture) is trained like an **energy-based
model** — low energy when a predicted embedding matches the target embedding,
high when it doesn't. Don't reconstruct the grain of the photo; be right about
its *meaning*.

**How OptimumAI applies it**
- `world_models/jepa.py` implements the energy `E(x,y) = ‖g(f(x)) − f(y)‖²` and
  contrasts it, in the trace itself, with wasteful pixel-level reconstruction.
- More broadly: a good *explanation* is a world model of a computation. If you can
  predict each intermediate value before it appears, you understand the op. The
  `Trace` exists so you can build that predictive intuition.

## 3. Anthropic — simplicity, transparency, interpretability

Anthropic's [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
makes an engineering case that generalizes far beyond agents: **find the simplest
thing that works; add complexity only when it demonstrably helps; and prize
transparency — show the system's steps.** Their interpretability research
([Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html),
[Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemanticity/index.html))
takes the same value inward: a model you cannot inspect is a model you cannot
trust, so decompose its polysemantic neurons into interpretable features.

**How OptimumAI applies it**
- **Transparency by default.** `explain=True` doesn't bolt on an explanation after
  the fact — the *same* code path emits the trace. Showing the work is the point.
- **Simplicity.** One primitive (`Trace`) carries every explanation. One pattern
  (`op()` computes; `op(explain=True)` also renders) covers every module. No
  framework, no ceremony.
- **Interpretability as a first-class topic.** `interpretability/superposition.py`
  brings Anthropic's own research into the library, so "why is this neuron
  polysemantic?" is a runnable question.

---

## Design principles these imply

1. **One computation path.** Fast mode and teaching mode run the same math; there
   is no "explanation" that can drift from the truth.
2. **One trace primitive.** `Step` + `Trace` model every operation from a dot
   product to a transformer block. Learn it once.
3. **Progressive disclosure.** `ExplainLevel` (beginner → researcher) reveals more
   of the same computation — formulas and complexity appear as you're ready for
   them, never as a different story.
4. **Offline-first, correctness-checked.** The core needs no API key. Where it
   matters, we cross-check exact autograd against numeric finite differences, so
   the teaching material is provably right.
5. **Composable, not clever.** Attention is `matmul` + `softmax`. An MLP is
   `Value` composed. Complexity is earned by building up from atoms you've already
   seen work.

The goal is not a faster framework — PyTorch exists. The goal is **the moment it
clicks**: when `softmax(QKᵀ/√dₖ)·V` stops being a formula you've memorized and
becomes something you could have derived yourself.

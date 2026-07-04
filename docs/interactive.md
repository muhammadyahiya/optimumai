# Interactive & Explained (v1.2)

Prompt-engineering patterns, the augmented-RNN lineage from
[distill.pub](https://distill.pub/2016/augmented-rnns/), and self-contained
interactive playgrounds inspired by
[Transformer Explainer](https://poloclub.github.io/transformer-explainer/) and
[TensorFlow Playground](https://playground.tensorflow.org).

## Prompt engineering — `optimumai.prompting`

Each pattern builds the prompt step by step and explains *why* it works and how it
fails. CLI: `optimumai prompt <pattern>`.

- **zero-shot** — role + instruction + task, no examples
- **few-shot** — in-context learning from K exemplars
- **chain-of-thought** — elicit reasoning before the answer
- **react** — interleave Thought / Action / Observation with a tool
- **self-consistency** — sample N chains, majority-vote the answer
- **structured-output** — constrain to a validated JSON schema

::: optimumai.prompting

## Augmented RNNs — `optimumai.augmented_rnns`

The pre-transformer ideas that made attention mainstream. CLI: `optimumai augrnn
attention|ntm|act`.

- **attention** — content-based soft attention as a differentiable memory read
- **ntm** — a Neural Turing Machine head: cosine-addressed read + erase/add write
- **act** — Adaptive Computation Time: a halting probability and ponder cost

::: optimumai.augmented_rnns

## Interactive playgrounds — `optimumai.visualization.playgrounds`

Self-contained HTML files with inline vanilla JS — **no server, no build, works
offline**. Generate one with `optimumai playground <name>`:

- **attention** — a Transformer-Explainer-style attention heatmap; hover a query
  token, drag a temperature slider to re-softmax the scores live
- **kmeans** — click to add points, watch Lloyd's algorithm iterate
- **astar** — draw walls on a grid, watch A\* expand the frontier to the goal

::: optimumai.visualization.playgrounds

## Concept gallery — `optimumai.visualization.gallery`

Per-concept matplotlib plots and animated GIFs, also reachable through the
`optimumai visualize <concept> --fmt png|gif` registry (needs the `[viz]` extra).

::: optimumai.visualization.gallery

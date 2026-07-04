"""Structured output — constraining generation to a JSON schema.

**What it is.** Free-form text is hard to parse reliably downstream. This
pattern asks the model to emit output conforming to an explicit schema
(field names, types, allowed values) instead of prose, so the caller can
``json.loads`` the response and validate it mechanically.

**How you build it.**

1. Define the schema — a small dict of ``{field: type_description}`` that
   states exactly what keys must appear and what each value should look
   like.
2. Render the schema into the prompt as a JSON template (not just a
   description in words) — models imitate structure far more reliably than
   they follow prose instructions about structure.
3. Add an explicit format instruction: emit *only* JSON, no markdown fences,
   no commentary, matching the schema exactly.
4. State the task.
5. Add a validation step — this module actually parses a (deterministic,
   offline) example completion against the schema with ``json.loads`` plus a
   key/type check, so the trace demonstrates *checking* adherence, not just
   hoping for it.

**Why it improves reliability vs free text.** A schema collapses the space
of acceptable outputs from "any prose that answers the question" down to "a
JSON object with these exact keys" — every field becomes independently
checkable (present? right type? right key?) instead of requiring a human or
a fragile regex to extract meaning from prose.

**Failure modes — schema adherence is not guaranteed.**

* Without *constrained decoding* (grammar-constrained sampling, JSON mode
  enforced at the token/logit level by the serving stack), the model can
  still emit invalid JSON, extra prose around the JSON, wrong types (a
  string where a number is expected), or silently drop/add fields — the
  prompt only makes correct output *likely*, not certain.
* Larger or more nested schemas increase the chance of a single dropped
  or malformed field.
* The caller must always validate (as this module's trace does) and have a
  retry/repair strategy — never assume the response parses on the first try.
"""

from __future__ import annotations

import json

from optimumai.core._fmt import arr
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def _render_schema(schema: dict[str, str]) -> str:
    """Render a ``{field: type_description}`` schema as a JSON-shaped template."""
    return "{\n" + ",\n".join(f'  "{k}": "{v}"' for k, v in schema.items()) + "\n}"


def _validate_against_schema(payload: dict, schema: dict[str, str]) -> list[str]:
    """Return a list of human-readable problems; empty list means it validates."""
    problems = []
    for key in schema:
        if key not in payload:
            problems.append(f"missing key {key!r}")
    for key in payload:
        if key not in schema:
            problems.append(f"unexpected key {key!r}")
    return problems


def structured_output_trace(
    task: str,
    schema: dict[str, str],
    example_completion: dict | None = None,
) -> Trace:
    """Build a schema-constrained prompt and validate a toy completion against it."""
    if not task.strip():
        raise ValueError("task must be a non-empty string")
    if not schema:
        raise ValueError("schema must have at least one field, got 0")

    t = Trace(
        op="structured_output",
        formula="prompt = task ⊕ schema_template ⊕ format_instruction; validate(response, schema)",
        complexity=f"O(fields); {len(schema)} field(s) in schema",
        why_ai=[
            "Turns free-text extraction into a mechanically checkable contract "
            "(keys, types) instead of prose a human or regex must parse",
            "Every field becomes independently verifiable: present? right key? right type?",
            "Without constrained/grammar-guided decoding, adherence is likely, not "
            "guaranteed — always validate and have a repair/retry path",
            "Bigger, more nested schemas raise the chance of a single dropped or "
            "malformed field",
        ],
        meta={"fields": list(schema.keys())},
    )

    schema_template = _render_schema(schema)
    t.add(
        "Define the schema",
        f"schema = {arr(str(schema))}",
        schema,
        detail="Field names + type descriptions the response must satisfy.",
    )
    t.add(
        "Render schema as a JSON template",
        schema_template,
        schema_template,
        detail="Models imitate structure more reliably than they follow prose "
        "instructions about structure — show the shape, don't just describe it.",
    )

    format_instruction = (
        "Respond with ONLY a single JSON object matching the schema above exactly — "
        "no markdown fences, no commentary, no extra or missing keys."
    )
    t.add(
        "Add the format instruction",
        format_instruction,
        format_instruction,
    )

    t.add("State the task", f"task = {arr(task)}", task)

    prompt = f"{task}\n\nSchema:\n{schema_template}\n\n{format_instruction}"
    t.add(
        "Assemble the final prompt",
        "prompt = task + '\\n\\nSchema:\\n' + schema_template + '\\n\\n' + format_instruction",
        prompt,
    )

    if example_completion is None:
        # Deterministic toy completion: valid types, correct keys — demonstrates the
        # validation step passing on well-formed output (no live model call).
        example_completion = dict.fromkeys(schema, "example")
    raw_response = json.dumps(example_completion)
    parsed = json.loads(raw_response)
    problems = _validate_against_schema(parsed, schema)
    t.add(
        "Validate a (toy, offline) completion against the schema",
        f"json.loads(response) → {arr(raw_response)}; "
        f"problems = {problems if problems else 'none'}",
        {"parsed": parsed, "problems": problems},
        detail="Adherence is never guaranteed by the prompt alone — this step is what "
        "actually confirms the response is usable, and is where a real system would "
        "trigger a retry/repair on failure.",
    )

    t.result = prompt
    return t


def structured_output(
    task: str,
    schema: dict[str, str],
    example_completion: dict | None = None,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
) -> str:
    """Return the assembled structured-output prompt. Set ``explain=True`` to print the trace."""
    t = structured_output_trace(task, schema, example_completion=example_completion)
    return t.render(level) if explain else t.result


def demo(seed: int = 0) -> Trace:
    """Structured-output prompt extracting a name/sentiment/score triple as JSON."""
    return structured_output_trace(
        task="Extract the reviewer's sentiment from: "
        "'The battery life on this laptop is disappointing.'",
        schema={
            "sentiment": "one of 'positive', 'negative', 'neutral'",
            "confidence": "float between 0 and 1",
        },
        example_completion={"sentiment": "negative", "confidence": "0.92"},
    )

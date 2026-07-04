import pytest

from optimumai.prompting import (
    chain_of_thought,
    chain_of_thought_trace,
    few_shot,
    few_shot_trace,
    react,
    react_trace,
    self_consistency,
    self_consistency_trace,
    structured_output,
    structured_output_trace,
    zero_shot,
    zero_shot_trace,
)

# ---------------------------------------------------------------------------
# zero_shot
# ---------------------------------------------------------------------------


def test_zero_shot_prompt_contains_role_instruction_task():
    prompt = zero_shot("Translate 'hello' to French.", instruction="Translate the text.")
    assert "Translate the text." in prompt
    assert "Translate 'hello' to French." in prompt
    assert "helpful" in prompt.lower()


def test_zero_shot_trace_shape():
    t = zero_shot_trace("Do the task.")
    assert len(t) == 4  # role, instruction, task, assemble
    assert isinstance(t.result, str)
    assert t.why_ai
    assert t.formula


def test_zero_shot_rejects_empty_task():
    with pytest.raises(ValueError):
        zero_shot_trace("")


def test_zero_shot_rejects_empty_instruction():
    with pytest.raises(ValueError):
        zero_shot_trace("task", instruction="  ")


# ---------------------------------------------------------------------------
# few_shot
# ---------------------------------------------------------------------------


def test_few_shot_prompt_contains_each_exemplar_and_query():
    examples = [("2+2", "4"), ("3+3", "6")]
    prompt = few_shot("5+5", examples)
    assert "Input: 2+2" in prompt
    assert "Output: 4" in prompt
    assert "Input: 3+3" in prompt
    assert "Input: 5+5" in prompt
    assert prompt.strip().endswith("Output:")


def test_few_shot_trace_step_count_scales_with_k():
    examples = [("a", "1"), ("b", "2"), ("c", "3")]
    t = few_shot_trace("d", examples)
    # 3 exemplars + query block + assemble = 5 (no instruction supplied)
    assert len(t) == 5
    assert isinstance(t.result, str)
    assert t.why_ai
    assert t.formula


def test_few_shot_rejects_zero_examples():
    with pytest.raises(ValueError):
        few_shot_trace("task", [])


def test_few_shot_rejects_empty_task():
    with pytest.raises(ValueError):
        few_shot_trace("", [("a", "1")])


# ---------------------------------------------------------------------------
# chain_of_thought
# ---------------------------------------------------------------------------


def test_chain_of_thought_zero_shot_includes_trigger():
    prompt = chain_of_thought("What is 12 * 4?")
    assert "Let's think step by step." in prompt
    assert "What is 12 * 4?" in prompt


def test_chain_of_thought_few_shot_includes_worked_exemplars():
    examples = [("2+2?", "2+2 is 4. Answer: 4")]
    prompt = chain_of_thought("3+3?", examples=examples)
    assert "2+2 is 4. Answer: 4" in prompt
    assert "Let's think step by step." not in prompt


def test_chain_of_thought_trace_shape():
    t = chain_of_thought_trace("What is 12 * 4?")
    assert isinstance(t.result, str)
    assert t.why_ai
    assert t.formula
    assert len(t) >= 3


def test_chain_of_thought_rejects_empty_task():
    with pytest.raises(ValueError):
        chain_of_thought_trace("")


# ---------------------------------------------------------------------------
# react
# ---------------------------------------------------------------------------


def test_react_prompt_contains_tool_and_cycle():
    prompt = react("Where is the Eiffel Tower?", tool_name="search")
    assert "search[" in prompt
    assert "Thought:" in prompt
    assert "Action:" in prompt
    assert "Observation:" in prompt
    assert "Final Answer:" in prompt


def test_react_trace_shape():
    t = react_trace("Where is the Eiffel Tower?")
    assert isinstance(t.result, str)
    assert t.why_ai
    assert t.formula
    assert len(t) == 5


def test_react_rejects_empty_task():
    with pytest.raises(ValueError):
        react_trace("")


def test_react_rejects_empty_tool_name():
    with pytest.raises(ValueError):
        react_trace("task", tool_name=" ")


# ---------------------------------------------------------------------------
# self_consistency
# ---------------------------------------------------------------------------


def test_self_consistency_majority_vote_wins():
    winner = self_consistency("2+2?", sampled_answers=["4", "4", "5"])
    assert winner == "4"


def test_self_consistency_trace_shape_and_tally():
    t = self_consistency_trace("2+2?", sampled_answers=["4", "4", "5"])
    assert isinstance(t.result, str)
    assert t.result == "4"
    assert t.why_ai
    assert t.formula
    # shared prompt + 3 sampled paths + tally + winner = 6
    assert len(t) == 6


def test_self_consistency_rejects_zero_samples():
    with pytest.raises(ValueError):
        self_consistency_trace("task", sampled_answers=[])


def test_self_consistency_rejects_empty_task():
    with pytest.raises(ValueError):
        self_consistency_trace("")


def test_self_consistency_default_samples_are_deterministic():
    t1 = self_consistency_trace("2+2?")
    t2 = self_consistency_trace("2+2?")
    assert t1.result == t2.result


# ---------------------------------------------------------------------------
# structured_output
# ---------------------------------------------------------------------------


def test_structured_output_prompt_contains_schema_and_instruction():
    schema = {"sentiment": "one of 'positive', 'negative'"}
    prompt = structured_output("Classify this review.", schema)
    assert '"sentiment"' in prompt
    assert "JSON" in prompt
    assert "Classify this review." in prompt


def test_structured_output_trace_validates_completion():
    schema = {"sentiment": "string", "confidence": "float"}
    t = structured_output_trace(
        "Classify.", schema, example_completion={"sentiment": "positive", "confidence": "0.9"}
    )
    assert isinstance(t.result, str)
    assert t.why_ai
    assert t.formula
    assert any("Validate" in step.title for step in t)


def test_structured_output_detects_schema_mismatch():
    schema = {"sentiment": "string"}
    t = structured_output_trace(
        "Classify.", schema, example_completion={"sentiment": "positive", "extra": "oops"}
    )
    validate_step = next(step for step in t if "Validate" in step.title)
    assert "unexpected key" in validate_step.expression


def test_structured_output_rejects_empty_schema():
    with pytest.raises(ValueError):
        structured_output_trace("task", {})


def test_structured_output_rejects_empty_task():
    with pytest.raises(ValueError):
        structured_output_trace("", {"a": "string"})

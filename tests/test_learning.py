import pytest
from click.testing import CliRunner

from optimumai.cli.main import cli
from optimumai.progress import ProgressTracker
from optimumai.quiz.engine import Question, Quiz, all_questions, available_quizzes
from optimumai.review.scheduler import ReviewScheduler, ReviewState, sm2


# --- quiz engine --------------------------------------------------------------
def test_quiz_bank_is_populated():
    assert len(available_quizzes()) >= 15
    assert len(all_questions()) >= 30


def test_quiz_perfect_and_wrong_scores():
    quiz = Quiz("softmax")
    assert quiz.grade([q.answer for q in quiz.questions]).score == 1.0
    wrong = [(q.answer + 1) % len(q.choices) for q in quiz.questions]
    assert quiz.grade(wrong).score < 1.0


def test_quiz_answer_indices_valid():
    for q in all_questions():
        assert isinstance(q, Question)
        assert 0 <= q.answer < len(q.choices)
        assert len(q.choices) >= 2


def test_quiz_unknown_topic_raises():
    with pytest.raises((KeyError, ValueError)):
        Quiz("no-such-topic")


def test_quiz_grade_tolerates_short_answers():
    quiz = Quiz("dot")
    assert quiz.grade([]).score == 0.0  # missing answers scored wrong


# --- spaced repetition (SM-2) -------------------------------------------------
def test_sm2_success_grows_interval():
    s0 = ReviewState()
    s1 = sm2(s0, quality=5, now=0.0)
    assert s1.repetitions == 1 and s1.interval_days == 1.0
    s2 = sm2(s1, quality=5, now=0.0)
    assert s2.repetitions == 2 and s2.interval_days == 6.0
    s3 = sm2(s2, quality=5, now=0.0)
    assert s3.interval_days > 6.0  # grows by ease


def test_sm2_failure_resets():
    s = sm2(sm2(ReviewState(), 5, now=0.0), 1, now=0.0)
    assert s.repetitions == 0 and s.interval_days == 1.0


def test_sm2_rejects_bad_quality():
    with pytest.raises(ValueError):
        sm2(ReviewState(), quality=9)


def test_scheduler_due_and_record(tmp_path):
    tracker = ProgressTracker(tmp_path / "p.json")
    sched = ReviewScheduler(tracker)
    # never-reviewed lessons are due immediately
    assert sched.due(["softmax", "dot"]) == ["softmax", "dot"]
    sched.record("softmax", quality=5, now=0.0)
    # after a successful review it's no longer due now
    assert "softmax" not in sched.due(["softmax", "dot"], now=1.0)
    assert sched.next_due(["softmax", "dot"], now=1.0) == "dot"


# --- CLI ----------------------------------------------------------------------
def _run(tmp_path, *args, **kw):
    env = {"OPTIMUMAI_PROGRESS_PATH": str(tmp_path / "p.json")}
    return CliRunner().invoke(cli, list(args), env=env, **kw)


def test_cli_quiz_lists(tmp_path):
    result = _run(tmp_path, "quiz")
    assert result.exit_code == 0
    assert "softmax" in result.output


def test_cli_quiz_interactive(tmp_path):
    quiz = Quiz("dot")
    answers = "\n".join(str(q.answer + 1) for q in quiz.questions) + "\n"
    result = _run(tmp_path, "quiz", "dot", input=answers)
    assert result.exit_code == 0
    assert "Score:" in result.output
    assert "100%" in result.output


def test_cli_review_when_nothing_done(tmp_path):
    # fresh tracker: every quiz is "due", so review picks one and quizzes it
    quiz = Quiz(available_quizzes()[0])
    answers = "\n".join(str(q.answer + 1) for q in quiz.questions) + "\n"
    result = _run(tmp_path, "review", input=answers)
    assert result.exit_code == 0
    assert "Reviewing:" in result.output


def test_cli_search(tmp_path):
    result = _run(tmp_path, "search", "attention")
    assert result.exit_code == 0
    assert "attention" in result.output.lower()


def test_cli_start(tmp_path):
    result = _run(tmp_path, "start")
    assert result.exit_code == 0
    assert "Welcome" in result.output
    assert "32" in result.output  # the dot demo result

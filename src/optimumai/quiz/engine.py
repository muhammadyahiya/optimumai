"""A pure active-recall engine — the testing effect, in code.

Cognitive science is unambiguous: *retrieving* an answer from memory strengthens
it far more than rereading it (the "testing effect" roughly doubles long-term
retention). So OptimumAI does not just *show* you the math — it *tests* you on it.

This module is the engine only: it is pure Python + numpy, does no I/O, and never
prints. A :class:`Quiz` pulls hand-written :class:`Question` objects for one
lesson out of the bank in :mod:`optimumai.quiz.questions`, and :meth:`Quiz.grade`
turns a learner's answers into a :class:`QuizResult`. An interactive CLI runner
that actually talks to a human lives elsewhere and is built on top of this.

    >>> from optimumai.quiz.engine import Quiz
    >>> q = Quiz("softmax")
    >>> result = q.grade([qq.answer for qq in q.questions])
    >>> result.score
    1.0
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Question:
    """A single multiple-choice question tied to one curriculum lesson.

    ``answer`` is the index into ``choices`` of the correct option, and
    ``explanation`` is a one-sentence justification shown after answering.
    """

    prompt: str
    choices: list[str]
    answer: int
    explanation: str
    lesson_id: str


@dataclass
class QuizResult:
    """The outcome of grading one quiz attempt.

    ``per_question`` records, in order, whether each question was answered
    correctly, so a caller can highlight exactly what to revisit.
    """

    total: int
    correct: int
    per_question: list[bool]

    @property
    def score(self) -> float:
        """Fraction correct in ``[0.0, 1.0]`` (``0.0`` for an empty quiz)."""
        if self.total == 0:
            return 0.0
        return self.correct / self.total


class Quiz:
    """A gradeable quiz over the questions for a single lesson."""

    def __init__(self, lesson_id: str):
        # Imported lazily to avoid a circular import: questions.py imports
        # Question from this module.
        from optimumai.quiz.questions import QUESTIONS

        if lesson_id not in QUESTIONS:
            available = ", ".join(sorted(QUESTIONS))
            raise KeyError(
                f"no quiz for lesson {lesson_id!r}; available quizzes: {available}"
            )
        self.lesson_id = lesson_id
        self._questions: list[Question] = list(QUESTIONS[lesson_id])

    @property
    def questions(self) -> list[Question]:
        """The questions for this lesson, in bank order."""
        return list(self._questions)

    def grade(self, answers: list[int]) -> QuizResult:
        """Grade ``answers`` against the correct indices.

        A length mismatch is tolerated: any question without a corresponding
        answer (and any answer past the last question) is scored as wrong.
        """
        per_question: list[bool] = []
        for i, question in enumerate(self._questions):
            given = answers[i] if i < len(answers) else None
            per_question.append(given == question.answer)
        correct = sum(per_question)
        return QuizResult(
            total=len(self._questions),
            correct=correct,
            per_question=per_question,
        )

    def check(self, index: int, choice: int) -> bool:
        """Return whether ``choice`` is the correct option for question ``index``."""
        return self._questions[index].answer == choice


def available_quizzes() -> list[str]:
    """Sorted lesson ids that currently have at least one question in the bank."""
    from optimumai.quiz.questions import QUESTIONS

    return sorted(lesson_id for lesson_id, qs in QUESTIONS.items() if qs)


def all_questions() -> list[Question]:
    """A flat list of every question across all lessons (for spaced repetition)."""
    from optimumai.quiz.questions import QUESTIONS

    flat: list[Question] = []
    for lesson_id in sorted(QUESTIONS):
        flat.extend(QUESTIONS[lesson_id])
    return flat

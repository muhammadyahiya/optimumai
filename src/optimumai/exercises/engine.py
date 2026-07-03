"""Compute-the-answer exercises — active recall with real numbers.

Unlike the multiple-choice :mod:`optimumai.quiz`, these ask the learner to
*compute* a value and grade it against the correct answer with a tolerance. Do
the work, submit a number, get graded.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Exercise:
    """A single compute-the-value task."""

    id: str
    lesson_id: str
    prompt: str
    answer: float
    tolerance: float = 1e-6
    hint: str = ""
    explanation: str = ""


@dataclass
class ExerciseResult:
    """The graded outcome of a submitted answer."""

    correct: bool
    submitted: float
    expected: float
    error: float


class Workbook:
    """Serve and grade compute-the-value exercises (optionally for one lesson)."""

    def __init__(self, lesson_id: str | None = None):
        from optimumai.exercises.bank import EXERCISES

        self._all = list(EXERCISES)
        if lesson_id is not None:
            self._items = [e for e in self._all if e.lesson_id == lesson_id]
            if not self._items:
                raise KeyError(f"no exercises for lesson {lesson_id!r}")
        else:
            self._items = list(self._all)
        self._by_id = {e.id: e for e in self._items}

    @property
    def exercises(self) -> list[Exercise]:
        return list(self._items)

    def get(self, exercise_id: str) -> Exercise:
        if exercise_id not in self._by_id:
            raise KeyError(f"unknown exercise {exercise_id!r}")
        return self._by_id[exercise_id]

    def grade(self, exercise_id: str, value: float) -> ExerciseResult:
        """Grade a submitted value (absolute tolerance, relative fallback for large answers)."""
        ex = self.get(exercise_id)
        error = abs(float(value) - ex.answer)
        tol = max(ex.tolerance, ex.tolerance * abs(ex.answer))
        return ExerciseResult(
            correct=error <= tol, submitted=float(value), expected=ex.answer, error=error
        )

    def check(self, exercise_id: str, value: float) -> bool:
        return self.grade(exercise_id, value).correct


def all_exercises() -> list[Exercise]:
    from optimumai.exercises.bank import EXERCISES

    return list(EXERCISES)


def available_exercises() -> list[str]:
    """Lesson ids that have at least one exercise."""
    seen: list[str] = []
    for e in all_exercises():
        if e.lesson_id not in seen:
            seen.append(e.lesson_id)
    return seen

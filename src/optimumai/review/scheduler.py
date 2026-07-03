"""Spaced-repetition review scheduling (an SM-2 variant).

Cognitive science is clear: **active recall** (testing yourself) plus **spaced
repetition** (reviewing at expanding intervals) is the most effective combination
for durable memory. This module schedules *when* to review each lesson, based on
how well you recalled it, using the classic SuperMemo-2 algorithm.

A review is graded 0–5 (``quality``): <3 means you failed to recall it (reset),
≥3 means success (the interval grows). State (ease, interval, repetitions, and
the next due time) is persisted via
:class:`~optimumai.progress.tracker.ProgressTracker`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

_DAY = 86_400.0
_MIN_EASE = 1.3
_DEFAULT_EASE = 2.5


@dataclass
class ReviewState:
    """SM-2 state for one lesson."""

    ease: float = _DEFAULT_EASE
    interval_days: float = 0.0
    repetitions: int = 0
    due: float = 0.0  # unix timestamp; 0 = due now

    def to_dict(self) -> dict:
        return {
            "ease": self.ease,
            "interval_days": self.interval_days,
            "repetitions": self.repetitions,
            "due": self.due,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ReviewState:
        return cls(
            ease=data.get("ease", _DEFAULT_EASE),
            interval_days=data.get("interval_days", 0.0),
            repetitions=data.get("repetitions", 0),
            due=data.get("due", 0.0),
        )


def sm2(state: ReviewState, quality: int, now: float | None = None) -> ReviewState:
    """Apply one SM-2 update given a recall ``quality`` in 0–5."""
    if not 0 <= quality <= 5:
        raise ValueError(f"quality must be in 0..5, got {quality}")
    now = time.time() if now is None else now

    if quality < 3:
        # Failed recall: reset the schedule, keep (slightly reduced) ease.
        reps = 0
        interval = 1.0
    else:
        reps = state.repetitions + 1
        if reps == 1:
            interval = 1.0
        elif reps == 2:
            interval = 6.0
        else:
            interval = round(state.interval_days * state.ease, 2)

    # Update ease per the SM-2 formula, clamped to a sensible floor.
    ease = state.ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    ease = max(_MIN_EASE, ease)

    return ReviewState(
        ease=round(ease, 3),
        interval_days=interval,
        repetitions=reps,
        due=now + interval * _DAY,
    )


class ReviewScheduler:
    """Schedule and surface spaced-repetition reviews over a ``ProgressTracker``."""

    def __init__(self, tracker):
        self.tracker = tracker

    def state_for(self, lesson_id: str) -> ReviewState:
        raw = self.tracker.review_state(lesson_id)
        return ReviewState.from_dict(raw) if raw else ReviewState()

    def record(self, lesson_id: str, quality: int, now: float | None = None) -> ReviewState:
        """Grade a review (0–5), persist the new schedule, and return it."""
        new_state = sm2(self.state_for(lesson_id), quality, now=now)
        self.tracker.set_review_state(lesson_id, new_state.to_dict())
        return new_state

    def due(self, candidates: list[str] | None = None, now: float | None = None) -> list[str]:
        """Lessons whose review is due (never-reviewed lessons are due immediately)."""
        now = time.time() if now is None else now
        states = self.tracker.all_review_states()
        pool = candidates if candidates is not None else list(states)
        out = []
        for lesson_id in pool:
            raw = states.get(lesson_id)
            if raw is None or raw.get("due", 0.0) <= now:
                out.append(lesson_id)
        return out

    def next_due(self, candidates: list[str] | None = None, now: float | None = None) -> str | None:
        due = self.due(candidates, now=now)
        return due[0] if due else None

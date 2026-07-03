"""Track a learner's progress through the OptimumAI course.

Progress is a tiny JSON file (``~/.optimumai/progress.json`` by default) recording
which lessons have been completed and when. Both the CLI and the Streamlit
dashboard read and write the same file, so your progress follows you across
sessions and surfaces.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def default_progress_path() -> Path:
    """Where progress is stored.

    Honors the ``OPTIMUMAI_PROGRESS_PATH`` environment variable (useful for tests
    and for keeping separate progress per project); otherwise
    ``~/.optimumai/progress.json``.
    """
    override = os.environ.get("OPTIMUMAI_PROGRESS_PATH")
    return Path(override) if override else Path.home() / ".optimumai" / "progress.json"


class ProgressTracker:
    """Records completed lessons and computes simple progress statistics."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else default_progress_path()
        self._data: dict = self._load()

    # ------------------------------------------------------------------ storage
    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {"completed": {}, "created": time.time()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))

    # ------------------------------------------------------------------- updates
    def mark_complete(self, lesson_id: str) -> None:
        self._data.setdefault("completed", {})[lesson_id] = time.time()
        self.save()

    def unmark(self, lesson_id: str) -> None:
        self._data.get("completed", {}).pop(lesson_id, None)
        self.save()

    def reset(self) -> None:
        self._data = {"completed": {}, "created": time.time()}
        self.save()

    # -------------------------------------------------------------------- queries
    def is_complete(self, lesson_id: str) -> bool:
        return lesson_id in self._data.get("completed", {})

    def completed_ids(self) -> set[str]:
        return set(self._data.get("completed", {}))

    def completed_count(self) -> int:
        return len(self._data.get("completed", {}))

    def completion_rate(self, total: int) -> float:
        """Fraction (0.0–1.0) of ``total`` lessons completed."""
        if total <= 0:
            return 0.0
        return min(1.0, self.completed_count() / total)

    # ---------------------------------------------------- spaced-repetition state
    def review_state(self, lesson_id: str) -> dict | None:
        """Return the stored SM-2 review state for a lesson, or None."""
        return self._data.get("reviews", {}).get(lesson_id)

    def set_review_state(self, lesson_id: str, state: dict) -> None:
        self._data.setdefault("reviews", {})[lesson_id] = state
        self.save()

    def all_review_states(self) -> dict[str, dict]:
        return dict(self._data.get("reviews", {}))

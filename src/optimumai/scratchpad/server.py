"""Local-first Flask app for the interactive math scratchpad.

The server binds 127.0.0.1, makes no external API calls, and does no maths --
boards recompute in the browser from a spec Python handed them. Its jobs are:
serve the page, resolve which front-end assets this board needs, report the
prerequisite graph's lock state, and record completion into the *course's*
progress store rather than a second one of its own.
"""

from __future__ import annotations

from flask import Flask, abort, jsonify, render_template

from optimumai.progress import ProgressTracker

from .assets import resolve_assets
from .concepts import get_concept, list_concepts
from .dag import unmet_prerequisites, validate_dag


def _completed_lessons(tracker: ProgressTracker) -> set[str]:
    return set(tracker.completed_ids())


def _completed_concepts(tracker: ProgressTracker) -> set[str]:
    """Concept ids whose *course lesson* is already marked complete.

    The mapping matters: progress is keyed by curriculum ``lesson_id``, so a
    lesson finished via ``optimumai learn`` shows as done here too, and vice
    versa. One store, two front doors.
    """
    done_lessons = _completed_lessons(tracker)
    return {c.concept_id for c in list_concepts() if c.lesson_id in done_lessons}


def create_app(progress_path: str | None = None) -> Flask:
    app = Flask(__name__)

    problems = validate_dag({c.concept_id: c for c in list_concepts()})
    if problems:  # fail loudly at startup, not silently in the sidebar
        raise ValueError("scratchpad prerequisite graph is invalid: " + "; ".join(problems))

    def tracker() -> ProgressTracker:
        return ProgressTracker(progress_path)

    @app.route("/")
    @app.route("/scratchpad/<concept_id>")
    def index(concept_id: str | None = None):
        concepts = list_concepts()
        concept_id = concept_id or concepts[0].concept_id
        try:
            active = get_concept(concept_id)
        except KeyError:
            abort(404)

        done = _completed_concepts(tracker())
        nav = []
        for c in concepts:
            unmet = unmet_prerequisites(c, done)
            nav.append({
                "concept_id": c.concept_id,
                "title": c.title,
                "locked": bool(unmet),
                "unmet": unmet,
                "completed": c.concept_id in done,
            })

        payload = active.to_dict()
        payload["unmet"] = unmet_prerequisites(active, done)
        payload["completed"] = active.concept_id in done
        return render_template(
            "index.html",
            active_concept=payload,
            all_concepts=nav,
            assets=resolve_assets(active.board.library, active.board.needs_katex),
        )

    @app.route("/api/concepts/<concept_id>")
    def api_concept(concept_id: str):
        try:
            concept = get_concept(concept_id)
        except KeyError:
            return jsonify({"error": f"unknown concept '{concept_id}'"}), 404
        return jsonify(concept.to_dict())

    @app.route("/api/concepts")
    def api_concepts():
        return jsonify([c.to_dict() for c in list_concepts()])

    @app.route("/api/order")
    def api_order():
        """The prerequisite graph, so the ordering rule is inspectable."""
        done = _completed_concepts(tracker())
        return jsonify([
            {
                "concept_id": c.concept_id,
                "lesson_id": c.lesson_id,
                "prerequisites": c.prerequisites,
                "unmet": unmet_prerequisites(c, done),
                "completed": c.concept_id in done,
            }
            for c in list_concepts()
        ])

    @app.route("/api/complete/<concept_id>", methods=["POST"])
    def api_complete(concept_id: str):
        try:
            concept = get_concept(concept_id)
        except KeyError:
            return jsonify({"error": f"unknown concept '{concept_id}'"}), 404
        t = tracker()
        t.mark_complete(concept.lesson_id)
        t.save()
        return jsonify({
            "completed": True,
            "concept_id": concept.concept_id,
            "lesson_id": concept.lesson_id,
        })

    return app

"""
optimumai.scratchpad.server
----------------------------
Local-first Flask app for the interactive math scratchpad. No external API
calls, no telemetry, no cloud dependency — this matches the local-first
design principle already used by optimumai-connect.

All heavy interactivity (dragging points, live recompute) happens in the
browser via JSXGraph. The server's only job is to serve the page and hand
over each concept's static metadata (why-AI-uses-it text, quiz, CLI hook).
"""

from dataclasses import asdict

from flask import Flask, abort, jsonify, render_template

from .concepts import get_concept, list_concepts


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    @app.route("/scratchpad/<concept_id>")
    def index(concept_id: str = "dot_product"):
        try:
            active = get_concept(concept_id)
        except KeyError:
            abort(404)
        return render_template(
            "index.html",
            active_concept=active,
            all_concepts=list_concepts(),
        )

    @app.route("/api/concepts/<concept_id>")
    def api_concept(concept_id: str):
        try:
            concept = get_concept(concept_id)
        except KeyError:
            return jsonify({"error": f"unknown concept '{concept_id}'"}), 404
        return jsonify(asdict(concept))

    @app.route("/api/concepts")
    def api_concepts():
        return jsonify([asdict(c) for c in list_concepts()])

    return app

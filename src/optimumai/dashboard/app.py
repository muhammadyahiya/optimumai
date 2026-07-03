"""OptimumAI Streamlit dashboard — browse the course, run lessons, track progress.

Launch with::

    optimumai dashboard          # or:  streamlit run <this file>

Progress is stored via :class:`~optimumai.progress.tracker.ProgressTracker`, so
what you complete here also shows up in ``optimumai progress`` on the CLI.
"""

from __future__ import annotations

import streamlit as st

from optimumai import __version__
from optimumai.core._fmt import arr
from optimumai.curriculum import COURSE
from optimumai.progress import ProgressTracker

_LEVELS = ["beginner", "intermediate", "engineer", "researcher"]


def _render_trace(trace, level: str = "engineer") -> None:
    """Render an OptimumAI Trace as Streamlit elements, gated by ``level``."""
    rank = _LEVELS.index(level)
    if trace.formula:
        st.markdown(f"**Formula:** `{trace.formula}`")
    show_detail = rank >= _LEVELS.index("intermediate")
    rows = []
    for s in trace.steps:
        row = {"#": s.index, "Step": s.title, "Computation": s.expression}
        if show_detail and s.detail:
            row["Notes"] = s.detail
        rows.append(row)
    if rows:
        st.table(rows)
    if trace.result is not None:
        st.markdown("**Result**")
        st.code(arr(trace.result))
    if trace.why_ai:
        st.info("**Why AI uses this**\n\n" + "\n".join(f"- {w}" for w in trace.why_ai))
    if trace.complexity and rank >= _LEVELS.index("engineer"):
        st.caption(f"Complexity: {trace.complexity}")


def _render_circuit_playground() -> None:
    """A live computation-graph 'circuit' from a user expression (v0.7)."""
    import os
    import tempfile

    import streamlit.components.v1 as components

    from optimumai.circuit.graph import build_from_expression
    from optimumai.circuit.render import to_html

    with st.expander("🔌 Circuit playground — see your own expression flow"):
        expr = st.text_input("Expression (variables default to 1)", "(a*b + c) * f")
        vars_str = st.text_input("Values", "a=2, b=-3, c=10, f=-2")
        if st.button("Build circuit"):
            variables: dict[str, float] = {}
            for pair in vars_str.split(","):
                key, _, val = pair.partition("=")
                if key.strip() and val.strip():
                    try:
                        variables[key.strip()] = float(val)
                    except ValueError:
                        pass
            try:
                _, graph = build_from_expression(expr, variables or None)
            except ValueError as exc:
                st.error(str(exc))
                return
            path = to_html(graph, os.path.join(tempfile.mkdtemp(), "circuit.html"))
            with open(path) as fh:
                components.html(fh.read(), height=480, scrolling=True)
            st.caption("Blue = data (forward) · Orange = grad (backward)")


def main() -> None:
    st.set_page_config(page_title="OptimumAI", page_icon="🧮", layout="wide")
    tracker = ProgressTracker()

    st.title("🧮 OptimumAI — the AI learning path")
    st.caption(f"Unlock the math behind AI, one step at a time · v{__version__}")

    total = len(COURSE)
    done = tracker.completed_ids()

    # --- Sidebar: progress + navigation ----------------------------------
    with st.sidebar:
        st.header("Your progress")
        st.progress(tracker.completion_rate(total))
        st.metric("Lessons completed", f"{len(done)} / {total}")
        nxt = COURSE.next_incomplete(done)
        if nxt:
            st.info(f"**Next up:** {nxt.title}")
        else:
            st.success("🎉 Course complete!")
        if st.button("Reset progress"):
            tracker.reset()
            st.rerun()

        st.divider()
        track_names = list(COURSE.tracks())
        chosen = st.radio("Jump to a track", ["All tracks", *track_names])

    # --- Main: tracks and lessons ----------------------------------------
    tracks = COURSE.tracks()
    level = st.select_slider("Explanation level", options=_LEVELS, value="engineer")

    _render_circuit_playground()

    for track_name, lessons in tracks.items():
        if chosen != "All tracks" and chosen != track_name:
            continue
        st.subheader(track_name)
        for lesson in lessons:
            complete = lesson.id in tracker.completed_ids()
            mark = "✅" if complete else "⬜"
            with st.expander(f"{mark}  {lesson.title} — {lesson.summary}"):
                cols = st.columns([1, 1, 4])
                if cols[0].button("▶ Run", key=f"run_{lesson.id}"):
                    st.session_state[f"ran_{lesson.id}"] = True
                if cols[1].button(
                    "Mark done" if not complete else "Undo",
                    key=f"done_{lesson.id}",
                ):
                    if complete:
                        tracker.unmark(lesson.id)
                    else:
                        tracker.mark_complete(lesson.id)
                    st.rerun()
                if lesson.prerequisites:
                    cols[2].caption("Prereqs: " + ", ".join(lesson.prerequisites))

                if st.session_state.get(f"ran_{lesson.id}") and lesson.demo:
                    _render_trace(lesson.demo(), level)


if __name__ == "__main__":
    main()

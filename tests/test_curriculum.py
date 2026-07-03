import pytest

from optimumai.core.trace import Trace
from optimumai.curriculum import COURSE, Course, Lesson


def test_course_has_lessons_with_unique_ids():
    ids = COURSE.ids()
    assert len(ids) == len(set(ids))
    assert len(COURSE) >= 15


def test_every_lesson_runs_and_returns_a_trace():
    for lesson in COURSE:
        assert lesson.demo is not None, lesson.id
        trace = lesson.demo()
        assert isinstance(trace, Trace), lesson.id
        assert len(trace) >= 1, lesson.id


def test_prerequisites_reference_real_lessons():
    ids = set(COURSE.ids())
    for lesson in COURSE:
        for prereq in lesson.prerequisites:
            assert prereq in ids, f"{lesson.id} -> missing prereq {prereq}"


def test_prerequisites_come_earlier_in_order():
    seen: set[str] = set()
    for lesson in COURSE:
        for prereq in lesson.prerequisites:
            assert prereq in seen, f"{lesson.id} needs {prereq} which comes later"
        seen.add(lesson.id)


def test_tracks_group_and_preserve_order():
    tracks = COURSE.tracks()
    assert len(tracks) >= 5
    assert sum(len(v) for v in tracks.values()) == len(COURSE)


def test_get_and_missing():
    assert COURSE.get("dot").id == "dot"
    with pytest.raises(KeyError):
        COURSE.get("nope")


def test_next_incomplete():
    assert COURSE.next_incomplete(set()).id == COURSE.ids()[0]
    assert COURSE.next_incomplete(set(COURSE.ids())) is None


def test_lesson_run_renders(capsys):
    trace = COURSE.get("dot").run("beginner")
    assert isinstance(trace, Trace)
    assert "DOT" in capsys.readouterr().out


def test_empty_lesson_run_returns_none():
    assert Lesson("x", "X", "T", "s", demo=None).run() is None


def test_default_course_instance_is_a_course():
    assert isinstance(COURSE, Course)

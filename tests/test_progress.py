from optimumai.progress import ProgressTracker


def _tracker(tmp_path):
    return ProgressTracker(tmp_path / "progress.json")


def test_mark_and_query(tmp_path):
    t = _tracker(tmp_path)
    assert not t.is_complete("dot")
    t.mark_complete("dot")
    assert t.is_complete("dot")
    assert t.completed_count() == 1
    assert "dot" in t.completed_ids()


def test_unmark_and_reset(tmp_path):
    t = _tracker(tmp_path)
    t.mark_complete("dot")
    t.mark_complete("cosine")
    t.unmark("dot")
    assert not t.is_complete("dot") and t.is_complete("cosine")
    t.reset()
    assert t.completed_count() == 0


def test_completion_rate(tmp_path):
    t = _tracker(tmp_path)
    assert t.completion_rate(10) == 0.0
    t.mark_complete("a")
    t.mark_complete("b")
    assert t.completion_rate(10) == 0.2
    assert t.completion_rate(0) == 0.0  # no divide-by-zero


def test_persistence_across_instances(tmp_path):
    _tracker(tmp_path).mark_complete("dot")
    assert _tracker(tmp_path).is_complete("dot")  # reloaded from disk


def test_corrupt_file_is_tolerated(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text("{ not valid json")
    t = ProgressTracker(path)
    assert t.completed_count() == 0  # falls back to empty, no crash


def test_env_override(tmp_path, monkeypatch):
    from optimumai.progress.tracker import default_progress_path

    target = tmp_path / "custom.json"
    monkeypatch.setenv("OPTIMUMAI_PROGRESS_PATH", str(target))
    assert default_progress_path() == target

from click.testing import CliRunner

from optimumai.cli.main import cli


def _run(tmp_path, *args):
    # redirect progress to a temp file so tests never touch real ~/.optimumai
    env = {"OPTIMUMAI_PROGRESS_PATH": str(tmp_path / "progress.json")}
    return CliRunner().invoke(cli, list(args), env=env)


def test_course_lists_tracks(tmp_path):
    result = _run(tmp_path, "course")
    assert result.exit_code == 0
    assert "learning path" in result.output.lower()
    assert "dot" in result.output


def test_learn_runs_and_marks_progress(tmp_path):
    result = _run(tmp_path, "learn", "dot")
    assert result.exit_code == 0
    assert "DOT" in result.output
    assert "complete" in result.output.lower()
    # progress should now show 1 completed
    progress = _run(tmp_path, "progress")
    assert "1/" in progress.output


def test_learn_no_track_does_not_record(tmp_path):
    _run(tmp_path, "learn", "dot", "--no-track")
    progress = _run(tmp_path, "progress")
    assert "0/" in progress.output


def test_learn_unknown_topic_errors(tmp_path):
    result = _run(tmp_path, "learn", "quantum-gravity")
    assert result.exit_code != 0


def test_progress_reset(tmp_path):
    _run(tmp_path, "learn", "dot")
    result = _run(tmp_path, "progress", "--reset")
    assert result.exit_code == 0
    assert "reset" in result.output.lower()
    assert "0/" in _run(tmp_path, "progress").output


def test_ask_offline(tmp_path):
    result = _run(tmp_path, "ask", "why softmax?")
    assert result.exit_code == 0
    assert len(result.output) > 0

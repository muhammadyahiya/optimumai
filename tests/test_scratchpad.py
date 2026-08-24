import pytest
from click.testing import CliRunner

from optimumai.cli.main import cli
from optimumai.scratchpad.concepts import get_concept, list_concepts

pytest.importorskip("flask")  # the scratchpad server needs the [scratchpad] extra

from optimumai.scratchpad.server import create_app  # noqa: E402


def test_list_concepts_nonempty():
    assert len(list_concepts()) >= 2


def test_get_concept_valid():
    c = get_concept("dot_product")
    # Tier 2: a concept declares a BoardSpec rather than naming a JS function.
    assert c.board.kind == "vectors"
    assert c.lesson_id == "dot"


def test_get_concept_invalid_raises():
    with pytest.raises(KeyError):
        get_concept("not_a_real_concept")


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_index_route(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"scratchpad" in resp.data.lower()


def test_api_concept_route(client):
    resp = client.get("/api/concepts/dot_product")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["concept_id"] == "dot_product"


def test_api_concept_404(client):
    resp = client.get("/api/concepts/nope")
    assert resp.status_code == 404


def test_scratchpad_page_404_for_unknown_concept(client):
    assert client.get("/scratchpad/nope").status_code == 404


def test_index_renders_board_hook_for_active_concept(client):
    """The template must hand the board type's concept id to scratchpad.js."""
    resp = client.get("/scratchpad/tangent_line")
    assert resp.status_code == 200
    assert b'window.ACTIVE_CONCEPT = "tangent_line"' in resp.data


# --- CLI wiring -----------------------------------------------------------


def test_cli_lists_boards_with_no_argument():
    result = CliRunner().invoke(cli, ["scratchpad"])
    assert result.exit_code == 0
    assert "dot_product" in result.output
    assert "tangent_line" in result.output


def test_cli_rejects_unknown_concept_without_binding_a_port():
    result = CliRunner().invoke(cli, ["scratchpad", "not_a_real_concept"])
    assert result.exit_code != 0
    assert "Unknown scratchpad concept" in result.output

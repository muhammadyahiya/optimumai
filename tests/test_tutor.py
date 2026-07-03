from optimumai.tutor import Tutor


def test_tutor_offline_does_not_crash():
    # litellm is not installed in the test env → tutor must degrade gracefully
    tutor = Tutor()
    assert tutor.available is False
    msg = tutor.ask("why softmax?")
    assert isinstance(msg, str) and len(msg) > 0


def test_tutor_explain_returns_string():
    msg = Tutor().explain("attention", level="beginner")
    assert isinstance(msg, str) and len(msg) > 0


def test_tutor_offline_message_is_helpful():
    msg = Tutor().ask("anything")
    assert "optimumai[llm]" in msg or "OPTIMUMAI_API_KEY" in msg

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_app_starts_and_renders_first_page():
    at = AppTest.from_file(APP).run(timeout=90)
    assert not at.exception
    assert any("Single customer" in h.value for h in at.header)


def test_predict_button_runs():
    at = AppTest.from_file(APP).run(timeout=90)
    at.button[0].click().run(timeout=90)
    assert not at.exception
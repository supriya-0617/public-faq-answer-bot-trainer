"""These pass on day one. They check the plumbing so you notice quickly if you break it."""
import py_compile

from app import llm
from app.storage import Store


def test_the_store_keeps_what_you_put_in_it(tmp_path):
    store = Store(str(tmp_path / "app.json"))
    store.put("hello", {"n": 1})
    assert store.get("hello") == {"n": 1}
    assert store.get("missing", "fallback") == "fallback"
    assert store.all() == {"hello": {"n": 1}}


def test_the_page_file_is_valid_python():
    py_compile.compile("app/main.py", doraise=True)


def test_missing_model_has_helpful_error(tmp_path):
    answer = llm.generate_answer("How do I apply?", tmp_path / "missing")
    assert "python -m app.train" in answer
    assert "python -m app.merge" in answer


def test_empty_question_is_rejected(tmp_path):
    try:
        llm.generate_answer("   ", tmp_path)
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("Empty question should fail")

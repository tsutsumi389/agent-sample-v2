"""app.agent の記憶整形ヘルパの単体テスト（DB・Ollama 不要の純関数）。"""

from types import SimpleNamespace

from app.agent import _format_items


def _item(value):
    """store の検索結果アイテムを模した最小オブジェクト（.value を持つ）。"""
    return SimpleNamespace(value=value)


def test_format_items_serializes_each_value_as_json_line():
    # langmem のエンベロープ（{"kind","content"}）をそのまま JSON で出す（剥がさない）
    items = [
        _item({"kind": "Memory", "content": {"content": "コーヒーが好き"}}),
        _item({"content": "本を読む"}),
    ]
    out = _format_items(items)
    lines = out.split("\n")
    assert lines[0] == '{"kind": "Memory", "content": {"content": "コーヒーが好き"}}'
    assert lines[1] == '{"content": "本を読む"}'


def test_format_items_keeps_japanese_unescaped():
    # ensure_ascii=False: 日本語が \uXXXX にエスケープされない
    out = _format_items([_item({"content": "田中"})])
    assert "田中" in out
    assert "\\u" not in out


def test_format_items_serializes_structured_profile():
    value = {
        "kind": "UserProfile",
        "content": {"name": "田中", "preferences": ["コーヒー", "読書"]},
    }
    out = _format_items([_item(value)])
    assert '"name": "田中"' in out
    assert '"preferences": ["コーヒー", "読書"]' in out


def test_format_items_skips_empty_values():
    # 値が空（None・空dict）のアイテムは行を作らない
    items = [_item(None), _item({}), _item({"content": "有効"})]
    out = _format_items(items)
    assert out == '{"content": "有効"}'


def test_format_items_empty():
    assert _format_items([]) == ""

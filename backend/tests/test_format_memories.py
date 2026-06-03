"""app.agent の記憶整形ヘルパの単体テスト（DB・Ollama 不要の純関数）。"""

from types import SimpleNamespace

from app.agent import _format_memories, _format_memory_value


def _item(value):
    """store の検索結果アイテムを模した最小オブジェクト（.value を持つ）。"""
    return SimpleNamespace(value=value)


def test_unwraps_kind_content_envelope_with_string():
    # langmem の一般記憶: {"kind": "Memory", "content": {"content": "..."}}
    value = {"kind": "Memory", "content": {"content": "コーヒーが好き"}}
    assert _format_memory_value(value) == "コーヒーが好き"


def test_handles_plain_content_dict():
    # 素の {"content": "..."} 形
    assert _format_memory_value({"content": "営業職"}) == "営業職"


def test_formats_structured_profile_dict_skipping_empty():
    # UserProfile の dump（エンベロープ込み）。空項目は除外される
    value = {
        "kind": "UserProfile",
        "content": {
            "name": "田中",
            "locale": None,
            "attributes": [],
            "preferences": ["コーヒー", "読書"],
            "response_style": "簡潔",
        },
    }
    out = _format_memory_value(value)
    assert "name: 田中" in out
    assert "preferences: コーヒー、読書" in out
    assert "response_style: 簡潔" in out
    assert "locale" not in out  # None は除外
    assert "attributes" not in out  # 空リストは除外


def test_handles_plain_string_value():
    assert _format_memory_value("そのままの文字列") == "そのままの文字列"


def test_format_memories_combines_profile_and_memories():
    profile = [_item({"kind": "UserProfile", "content": {"name": "田中"}})]
    memories = [
        _item({"kind": "Memory", "content": {"content": "コーヒーが好き"}}),
        _item({"content": "本を読む"}),
    ]
    out = _format_memories(profile, memories)
    lines = out.split("\n")
    assert lines[0] == "- [プロフィール] name: 田中"
    assert "- コーヒーが好き" in lines
    assert "- 本を読む" in lines


def test_format_memories_empty():
    assert _format_memories([], []) == ""

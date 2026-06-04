"""app.agent のグラフ構成と recall ノードの単体テスト（DB・Ollama 不要）。"""

import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from app.agent import SYSTEM_PROMPT_BASE, _build_system_text, build_agent, recall_node


def _item(value):
    """store の検索結果アイテムを模した最小オブジェクト（.value を持つ）。"""
    return SimpleNamespace(value=value)


class _StubStore:
    """asearch の呼び出しを記録し、namespace ごとに固定結果を返すスタブ。"""

    def __init__(self, results: dict):
        self.results = results
        self.calls: list[tuple] = []

    async def asearch(self, namespace, *, query=None, limit=10):
        self.calls.append((namespace, query, limit))
        return self.results.get(namespace, [])


def test_recall_node_injects_profile_and_memories():
    store = _StubStore(
        {
            ("profile", "u1"): [_item({"kind": "UserProfile", "content": {"name": "田中"}})],
            ("memories", "u1"): [_item({"kind": "Memory", "content": {"content": "コーヒーが好き"}})],
        }
    )
    state = {"messages": [HumanMessage("おすすめの飲み物は？")]}
    config = {"configurable": {"user_id": "u1"}}

    out = asyncio.run(recall_node(state, store=store, config=config))

    # プロフィールと一般記憶は別の state キーに、JSON のまま注入される
    assert out["recalled_profile"] == '{"kind": "UserProfile", "content": {"name": "田中"}}'
    assert (
        out["recalled_memories"]
        == '{"kind": "Memory", "content": {"content": "コーヒーが好き"}}'
    )
    # プロフィールは常時取得（query なし）、一般記憶のみ直近発話で意味検索
    assert store.calls == [
        (("profile", "u1"), None, 5),
        (("memories", "u1"), "おすすめの飲み物は？", 5),
    ]


def test_recall_node_uses_last_human_message_as_query():
    store = _StubStore({})
    state = {
        "messages": [
            HumanMessage("最初の質問"),
            AIMessage("回答です"),
            HumanMessage("2番目の質問"),
        ]
    }
    config = {"configurable": {"user_id": "u1"}}

    asyncio.run(recall_node(state, store=store, config=config))

    memory_calls = [c for c in store.calls if c[0] == ("memories", "u1")]
    assert all(query == "2番目の質問" for _, query, _ in memory_calls)


def test_recall_node_without_user_id_skips_search():
    store = _StubStore({})
    state = {"messages": [HumanMessage("こんにちは")]}
    config = {"configurable": {}}

    out = asyncio.run(recall_node(state, store=store, config=config))

    assert out == {"recalled_profile": "", "recalled_memories": ""}
    assert store.calls == []


def test_recall_node_falls_back_to_empty_on_store_error():
    # 想起は付加機能: store 検索が失敗しても会話を落とさず記憶なしで継続する
    class _BrokenStore:
        async def asearch(self, *args, **kwargs):
            raise RuntimeError("db down")

    state = {"messages": [HumanMessage("こんにちは")]}
    config = {"configurable": {"user_id": "u1"}}

    out = asyncio.run(recall_node(state, store=_BrokenStore(), config=config))

    assert out == {"recalled_profile": "", "recalled_memories": ""}


def test_build_system_text_without_memories():
    assert _build_system_text("", "") == SYSTEM_PROMPT_BASE


def test_build_system_text_injects_profile_and_memory_sections():
    out = _build_system_text('{"name": "田中"}', '{"content": "コーヒーが好き"}')
    assert out.startswith(SYSTEM_PROMPT_BASE)
    # プロフィール（常時）と記憶（話題依存）が XML タグで区切られて注入される
    assert '<user_profile>\n{"name": "田中"}\n</user_profile>' in out
    assert '<related_memories>\n{"content": "コーヒーが好き"}\n</related_memories>' in out


def test_build_system_text_profile_only():
    # SYSTEM_PROMPT_BASE の説明文にもタグ名が登場するため、注入セクションの有無は
    # 閉じタグで判定する
    out = _build_system_text('{"name": "田中"}', "")
    assert "</user_profile>" in out
    assert "</related_memories>" not in out


def test_build_agent_graph_structure():
    # グラフ構成: START → recall → agent ⇄ tools → END（コンパイルのみ、LLM 呼び出し無し）
    graph = build_agent(store=None, checkpointer=None)
    nodes = set(graph.get_graph().nodes)
    assert {"recall", "agent", "tools"} <= nodes

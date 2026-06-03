from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from langgraph.store.base import BaseStore
from langmem import create_manage_memory_tool, create_search_memory_tool

from app.llm import get_chat_model

# エージェントへの基本システム指示。記憶ツールの能動的な利用を促す（gemma4 のような
# ローカルモデルでもツール呼び出しを行いやすくするため明示する）。プロアクティブ想起
# （dynamic_prompt）が毎ターン記憶を注入するため、search_memory はフォールバック用途。
SYSTEM_PROMPT_BASE = """あなたは長期記憶を持つ親切な日本語アシスタントです。

記憶の扱い方:
- ユーザーに関する新しい事実（名前・好み・所属・予定など）を知ったら、
  manage_memory ツールで保存してください。
- 以下の「既知のユーザー情報」は過去の会話から想起した内容です。これを踏まえて
  パーソナライズした応答をしてください。情報が不足する場合のみ search_memory で補ってください。
- 応答スタイルの指定があれば必ず従ってください。
"""


def _format_memory_value(value: Any) -> str:
    """store のメモリ値を1行の読める文字列に整形する。

    langmem は値を {"kind": <SchemaName>, "content": <model_dump | {"content": str}>}
    のエンベロープで保存する。素の {"content": str} 形や任意 dict にも対応する。
    """
    # {"kind","content"} エンベロープと、一般記憶の二重ラップ {"content": str} を剥がす。
    # 構造化 dict（UserProfile 等、複数フィールド）はここで止めてそのまま整形する。
    content = value
    while isinstance(content, dict) and set(content.keys()) in ({"kind", "content"}, {"content"}):
        content = content["content"]

    if isinstance(content, dict):
        # UserProfile のような構造化 dict: 空でない項目だけを "key: 値" で連結
        parts = []
        for k, v in content.items():
            if v in (None, "", [], {}):
                continue
            v_str = "、".join(map(str, v)) if isinstance(v, list) else str(v)
            parts.append(f"{k}: {v_str}")
        return " / ".join(parts)
    return str(content)


def _format_memories(profile_items, memory_items) -> str:
    """プロフィールと一般記憶を、システムプロンプトに埋め込む箇条書きに整形する。"""
    lines: list[str] = []
    for item in profile_items:
        text = _format_memory_value(item.value)
        if text:
            lines.append(f"- [プロフィール] {text}")
    for item in memory_items:
        text = _format_memory_value(item.value)
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines)


def build_agent(store, checkpointer):
    """プロアクティブ記憶想起付きの ReAct エージェントを構築する。

    - store: 長期記憶（pgvector 意味検索付き）
    - checkpointer: 会話履歴（thread_id 単位の短期記憶）

    prompt は dynamic_prompt（async callable）。LangGraph の RunnableCallable が
    署名から store / config を実行時に注入する（引数名+型注釈で判定）。毎ターン直近
    の発話で意味検索し、関連記憶をシステムプロンプトへ確実に注入する。
    """

    async def dynamic_prompt(
        state, *, store: BaseStore, config: RunnableConfig
    ) -> list:
        user_id = config["configurable"].get("user_id")
        messages = state["messages"]

        sys_text = SYSTEM_PROMPT_BASE
        if user_id:
            # 直近のユーザー発話を意味検索クエリにする（無ければ通常検索にフォールバック）
            last_user = next(
                (m for m in reversed(messages) if getattr(m, "type", None) == "human"),
                None,
            )
            query = last_user.content if last_user else None

            profile_items = await store.asearch(("profile", user_id), query=query, limit=5)
            memory_items = await store.asearch(("memories", user_id), query=query, limit=5)

            block = _format_memories(profile_items, memory_items)
            if block:
                sys_text = f"{SYSTEM_PROMPT_BASE}\n\n# 既知のユーザー情報\n{block}"

        return [SystemMessage(content=sys_text)] + list(messages)

    return create_react_agent(
        get_chat_model(),
        prompt=dynamic_prompt,
        tools=[
            create_manage_memory_tool(namespace=("memories", "{user_id}")),
            create_search_memory_tool(namespace=("memories", "{user_id}")),
        ],
        store=store,
        checkpointer=checkpointer,
    )

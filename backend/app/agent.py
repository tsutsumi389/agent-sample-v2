import json

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.base import BaseStore
from langmem import create_manage_memory_tool, create_search_memory_tool

from app.llm import get_chat_model

# エージェントへの基本システム指示。記憶ツールの能動的な利用を促す（gemma4 のような
# ローカルモデルでもツール呼び出しを行いやすくするため明示する）。プロアクティブ想起
# （recall ノード）が毎ターン記憶を注入するため、search_memory はフォールバック用途。
SYSTEM_PROMPT_BASE = """あなたは長期記憶を持つ親切な日本語アシスタントです。

記憶の扱い方:
- ユーザーに関する新しい事実（名前・好み・所属・予定など）を知ったら、
  manage_memory ツールで保存してください。
- <user_profile> タグ内は確定済みのユーザー基本情報（JSON）です。常にこれを
  踏まえてパーソナライズした応答をしてください。
- <related_memories> タグ内は今の話題に関連して過去の会話から想起した記憶
  （JSON、1行1件）です。関連する場合のみ活用し、情報が不足する場合のみ
  search_memory で補ってください。
- 応答スタイルの指定があれば必ず従ってください。
"""


class AgentState(MessagesState):
    """recall ノードの想起結果を下流ノードへ受け渡すグラフ state。

    常時注入する基本情報（プロフィール）と、話題依存で厳選した記憶を分けて持つ。
    """

    recalled_profile: str
    recalled_memories: str


def _format_items(items) -> str:
    """store の検索結果アイテム列を、1行1件の JSON（JSONL）に整形する。

    langmem の保存形式（{"kind", "content"} エンベロープ等）をそのまま出すことで、
    形状判定による剥がしロジックを持たない。日本語が \\uXXXX にエスケープされない
    よう ensure_ascii=False を指定する。
    """
    return "\n".join(
        json.dumps(item.value, ensure_ascii=False) for item in items if item.value
    )


def _build_system_text(recalled_profile: str, recalled_memories: str) -> str:
    """想起済みのプロフィール・記憶ブロックをシステムプロンプトへ合成する。

    プロフィール（常時注入の確定情報）と話題依存の記憶を XML タグで区切り、
    LLM にデータの境界と確度の違いを伝える。
    """
    sys_text = SYSTEM_PROMPT_BASE
    if recalled_profile:
        sys_text += f"\n\n<user_profile>\n{recalled_profile}\n</user_profile>"
    if recalled_memories:
        sys_text += f"\n\n<related_memories>\n{recalled_memories}\n</related_memories>"
    return sys_text


async def recall_node(
    state: AgentState, *, store: BaseStore, config: RunnableConfig
) -> dict:
    """ターン冒頭に1回だけ長期記憶を取得し、結果を state に載せるノード。

    LangGraph がノード関数の署名（引数名+型注釈）から store / config を実行時に
    注入する。検索はターンあたり1回で、下流の agent ノードは state を読むだけに
    する（ReAct ループ中に再検索しない）。

    - プロフィール: 基本情報なので話題に関係なく常に取得する
    - 一般記憶: 直近のユーザー発話で意味検索し、話題に関連するものを厳選する
    """
    user_id = config["configurable"].get("user_id")
    if not user_id:
        return {"recalled_profile": "", "recalled_memories": ""}

    # 直近のユーザー発話を意味検索クエリにする（無ければ通常検索にフォールバック）
    last_user = next(
        (m for m in reversed(state["messages"]) if getattr(m, "type", None) == "human"),
        None,
    )
    query = last_user.content if last_user else None

    # プロアクティブ想起は付加機能。検索失敗（DB 一時断など）で会話全体を
    # 落とさず、記憶なしとして応答を継続する。
    try:
        # プロフィール = 常時注入: query なし（類似度ランキング非依存）で必ず取得する。
        # 単一の集約オブジェクト前提のため limit は安全弁（複数件できた場合の注入上限）。
        profile_items = await store.asearch(("profile", user_id), limit=5)
        # 一般記憶 = 話題依存: 意味検索で上位5件に厳選する
        memory_items = await store.asearch(("memories", user_id), query=query, limit=5)
    except Exception:  # noqa: BLE001
        return {"recalled_profile": "", "recalled_memories": ""}

    return {
        "recalled_profile": _format_items(profile_items),
        "recalled_memories": _format_items(memory_items),
    }


def build_agent(store, checkpointer):
    """プロアクティブ記憶想起付きのエージェントグラフを構築する。

    - store: 長期記憶（pgvector 意味検索付き）
    - checkpointer: 会話履歴（thread_id 単位の短期記憶）

    グラフ構成: START → recall → agent ⇄ tools → END
    - recall: ターン冒頭に1回だけ記憶を取得し state に載せる（プロアクティブ想起）。
      プロフィールは常時取得、一般記憶は話題で意味検索
    - agent: 想起結果をシステムプロンプトに注入して LLM を呼ぶ
    - tools: manage_memory / search_memory。ツール呼び出し後は agent へ戻る（ReAct ループ）
    """
    tools = [
        create_manage_memory_tool(namespace=("memories", "{user_id}")),
        create_search_memory_tool(namespace=("memories", "{user_id}")),
    ]
    llm = get_chat_model().bind_tools(tools)

    async def agent_node(state: AgentState) -> dict:
        """想起済みのプロフィール・記憶をシステムプロンプトへ注入して LLM を呼ぶノード。"""
        sys_text = _build_system_text(
            state.get("recalled_profile", ""), state.get("recalled_memories", "")
        )
        response = await llm.ainvoke(
            [SystemMessage(content=sys_text)] + state["messages"]
        )
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("recall", recall_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "recall")
    builder.add_edge("recall", "agent")
    # 最後の AI メッセージにツール呼び出しがあれば tools へ、無ければ END へ
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    return builder.compile(store=store, checkpointer=checkpointer)

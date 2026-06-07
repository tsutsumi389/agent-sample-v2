from langchain_core.messages import AIMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.store.base import BaseStore

from app.agent.formatting import _format_items, _message_text
from app.agent.state import AgentState


def finalize_node(state) -> dict:
    """合格（または上限到達）したドラフトを最終回答として会話履歴に確定する。

    ループ中の計画・評価・途中ドラフトは messages に入れないため、checkpointer に
    残る履歴と reflection（バックグラウンド記憶抽出）への入力は「ユーザー発話 +
    最終回答」だけの綺麗な会話になる。

    ドラフトが空（モデルが内容を返せなかった等）の場合は、空のバブルを
    ユーザーに見せないようフォールバック文言を返す。
    """
    draft = state.get("draft", "")
    if not draft:
        draft = "申し訳ありません。うまく回答を生成できませんでした。もう一度お試しください。"
    return {"messages": [AIMessage(content=draft)]}


async def recall_node(
    state: AgentState, *, store: BaseStore, config: RunnableConfig
) -> dict:
    """ターン冒頭に1回だけ長期記憶を取得し、結果を state に載せるノード。

    LangGraph がノード関数の署名（引数名+型注釈）から store / config を実行時に
    注入する。検索はターンあたり1回で、下流の Planner / Executor は state を読む
    だけにする（ReAct ループ中に再検索しない）。

    - プロフィール: 基本情報なので話題に関係なく常に取得する
    - 一般記憶: 直近のユーザー発話で意味検索し、話題に関連するものを厳選する

    あわせてループ制御フィールド（iteration 等）と、前ターンが異常終了した場合に
    残り得る scratch を初期化する。
    """
    # ループ制御の初期化。scratch は add_messages reducer のため REMOVE_ALL で消す。
    loop_init = {
        "plan": "",
        "draft": "",
        "evaluation": {},
        "feedback": "",
        "iteration": 0,
        "planner_scratch": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        "executor_scratch": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
    }

    user_id = config["configurable"].get("user_id")
    if not user_id:
        return {"recalled_profile": "", "recalled_memories": "", **loop_init}

    # 直近のユーザー発話を意味検索クエリにする（無ければ通常検索にフォールバック）
    last_user = next(
        (m for m in reversed(state["messages"]) if getattr(m, "type", None) == "human"),
        None,
    )
    # マルチパート content（list 形式）にも対応するため _message_text で文字列化する
    query = _message_text(last_user) if last_user else None

    # プロアクティブ想起は付加機能。検索失敗（DB 一時断など）で会話全体を
    # 落とさず、記憶なしとして応答を継続する。
    try:
        # プロフィール = 常時注入: query なし（類似度ランキング非依存）で必ず取得する。
        # 単一の集約オブジェクト前提のため limit は安全弁（複数件できた場合の注入上限）。
        profile_items = await store.asearch(("profile", user_id), limit=5)
        # 一般記憶 = 話題依存: 意味検索で上位5件に厳選する
        memory_items = await store.asearch(("memories", user_id), query=query, limit=5)
    except Exception:  # noqa: BLE001
        return {"recalled_profile": "", "recalled_memories": "", **loop_init}

    return {
        "recalled_profile": _format_items(profile_items),
        "recalled_memories": _format_items(memory_items),
        **loop_init,
    }

from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages


class AgentState(MessagesState):
    """Planner→Executor→Evaluator ループのグラフ state。

    messages（会話履歴 = checkpointer 永続対象）には最終回答のみを追加し、
    計画・ドラフト・評価などのループ内部情報は専用フィールドに分離して履歴を
    汚さない。Planner / Executor の ReAct ツール往復も scratch フィールド上で
    行い、本体 messages とは混ぜない。
    """

    recalled_profile: str
    recalled_memories: str
    plan: str  # Planner の最新計画
    draft: str  # Executor の最新ドラフト回答
    evaluation: dict  # {"verdict": "ok"|"ng", "score": int|None, "feedback": str}
    feedback: str  # NG 時に Planner へ渡す改善指示
    iteration: int  # 試行回数（Planner が計画を確定するたびに +1）
    planner_scratch: Annotated[list[AnyMessage], add_messages]
    executor_scratch: Annotated[list[AnyMessage], add_messages]

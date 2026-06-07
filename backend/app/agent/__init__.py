"""Planner→Executor→Evaluator ループのマルチエージェント実装。

責務別モジュール構成:
- prompts: システムプロンプト定数
- state: グラフ state（AgentState）
- formatting: 記憶・メッセージの整形ヘルパー
- evaluation: Evaluator 出力のパース
- routing: 条件エッジ（ループ分岐）
- nodes: recall / finalize ノード
- graph: グラフ構築（build_agent）

旧 `app.agent` モジュールからの import 互換のため、主要シンボルを再エクスポートする。
"""

from app.agent.evaluation import parse_evaluation
from app.agent.graph import build_agent
from app.agent.nodes import finalize_node, recall_node
from app.agent.prompts import (
    EVALUATOR_SYSTEM_PROMPT,
    EXECUTOR_SYSTEM_PROMPT,
    MEMORY_GUIDE,
    PLANNER_FEEDBACK_TEMPLATE,
    PLANNER_SYSTEM_PROMPT,
)
from app.agent.routing import executor_route, planner_route, route_after_eval
from app.agent.state import AgentState

__all__ = [
    "AgentState",
    "EVALUATOR_SYSTEM_PROMPT",
    "EXECUTOR_SYSTEM_PROMPT",
    "MEMORY_GUIDE",
    "PLANNER_FEEDBACK_TEMPLATE",
    "PLANNER_SYSTEM_PROMPT",
    "build_agent",
    "executor_route",
    "finalize_node",
    "parse_evaluation",
    "planner_route",
    "recall_node",
    "route_after_eval",
]

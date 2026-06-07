from langchain_core.messages import AIMessage, RemoveMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import ToolNode
from langmem import create_manage_memory_tool, create_search_memory_tool

from app.agent.evaluation import parse_evaluation
from app.agent.formatting import _build_system_text, _message_text
from app.agent.nodes import finalize_node, recall_node
from app.agent.prompts import (
    EVALUATOR_SYSTEM_PROMPT,
    EXECUTOR_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
)
from app.agent.routing import executor_route, planner_route, route_after_eval
from app.agent.state import AgentState
from app.config import settings
from app.llm import get_chat_model


def build_agent(store, checkpointer):
    """Planner→Executor→Evaluator ループ構成のマルチエージェントグラフを構築する。

    - store: 長期記憶（pgvector 意味検索付き）
    - checkpointer: 会話履歴（thread_id 単位の短期記憶）

    グラフ構成:
        START → recall → planner_agent ⇄ planner_tools
                              ↓
                         executor_agent ⇄ executor_tools
                              ↓
                         evaluator ─ ng（上限未満）→ planner_agent（再計画）
                              └─ ok / 上限到達 → finalize → END

    - recall: ターン冒頭に1回だけ記憶を取得し state に載せる（プロアクティブ想起）
    - planner_agent: 回答計画を立てる。記憶ツール使用可（planner_scratch 上で ReAct）
    - executor_agent: 計画に従い最終回答ドラフトを生成。記憶ツール使用可。
      SSE でトークンを配信する唯一のノード
    - evaluator: ドラフトを評価し、不合格ならフィードバック付きで Planner へ戻す
    - finalize: 確定ドラフトのみを messages（会話履歴）に追加する
    """
    tools = [
        create_manage_memory_tool(namespace=("memories", "{user_id}")),
        create_search_memory_tool(namespace=("memories", "{user_id}")),
    ]
    llm_with_tools = get_chat_model().bind_tools(tools)
    llm_plain = get_chat_model()

    def _pick_model(scratch):
        """ツール往復が上限に達したら、ツールなしモデルで強制的に最終出力させる。

        ローカルモデルがツールを呼び続ける暴走の安全弁。AIMessage の数 = これまでの
        ツール往復回数。
        """
        tool_turns = sum(1 for m in scratch if isinstance(m, AIMessage))
        return llm_with_tools if tool_turns < settings.MAX_TOOL_TURNS else llm_plain

    async def planner_agent(state: AgentState) -> dict:
        """ユーザー要求と記憶から回答計画を立てるノード（scratch 上で ReAct）。"""
        scratch = state.get("planner_scratch") or []
        sys_text = _build_system_text(
            PLANNER_SYSTEM_PROMPT,
            state.get("recalled_profile", ""),
            state.get("recalled_memories", ""),
        )
        feedback = state.get("feedback", "")
        if feedback:
            sys_text += (
                f"\n\n<evaluator_feedback>\n{feedback}\n</evaluator_feedback>\n"
                "前回の計画に基づく回答は上記の点で不十分と評価されました。"
                "フィードバックを反映した改善計画を立ててください。"
            )
        prompt = [SystemMessage(content=sys_text)] + state["messages"] + list(scratch)
        response = await _pick_model(scratch).ainvoke(prompt)
        if getattr(response, "tool_calls", None):
            return {"planner_scratch": [response]}
        return {
            "plan": _message_text(response),
            "iteration": state.get("iteration", 0) + 1,
            "planner_scratch": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        }

    async def executor_agent(state: AgentState) -> dict:
        """計画に従ってユーザーへの最終回答ドラフトを生成するノード（scratch 上で ReAct）。"""
        scratch = state.get("executor_scratch") or []
        sys_text = _build_system_text(
            EXECUTOR_SYSTEM_PROMPT,
            state.get("recalled_profile", ""),
            state.get("recalled_memories", ""),
        )
        sys_text += f"\n\n<plan>\n{state.get('plan', '')}\n</plan>"
        prompt = [SystemMessage(content=sys_text)] + state["messages"] + list(scratch)
        response = await _pick_model(scratch).ainvoke(prompt)
        if getattr(response, "tool_calls", None):
            return {"executor_scratch": [response]}
        return {
            "draft": _message_text(response),
            "executor_scratch": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        }

    async def evaluator_node(state: AgentState) -> dict:
        """計画とドラフトを評価し、合否とフィードバックを state に書くノード。"""
        sys_text = (
            EVALUATOR_SYSTEM_PROMPT
            + f"\n\n<plan>\n{state.get('plan', '')}\n</plan>"
            + f"\n\n<draft>\n{state.get('draft', '')}\n</draft>"
        )
        response = await llm_plain.ainvoke(
            [SystemMessage(content=sys_text)] + state["messages"]
        )
        evaluation = parse_evaluation(_message_text(response))
        feedback = evaluation["feedback"] if evaluation["verdict"] == "ng" else ""
        return {"evaluation": evaluation, "feedback": feedback}

    builder = StateGraph(AgentState)
    builder.add_node("recall", recall_node)
    builder.add_node("planner_agent", planner_agent)
    builder.add_node("planner_tools", ToolNode(tools, messages_key="planner_scratch"))
    builder.add_node("executor_agent", executor_agent)
    builder.add_node("executor_tools", ToolNode(tools, messages_key="executor_scratch"))
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "recall")
    builder.add_edge("recall", "planner_agent")
    # Planner の ReAct ループ: ツール呼び出しがあれば実行して戻る、なければ Executor へ
    builder.add_conditional_edges(
        "planner_agent",
        planner_route,
        {"tools": "planner_tools", "next": "executor_agent"},
    )
    builder.add_edge("planner_tools", "planner_agent")
    # Executor の ReAct ループ: 同上。完了したら Evaluator へ
    builder.add_conditional_edges(
        "executor_agent",
        executor_route,
        {"tools": "executor_tools", "next": "evaluator"},
    )
    builder.add_edge("executor_tools", "executor_agent")
    # 評価結果で分岐: 合格/上限到達 → finalize、不合格 → 再計画
    builder.add_conditional_edges(
        "evaluator",
        route_after_eval,
        {"finalize": "finalize", "planner_agent": "planner_agent"},
    )
    builder.add_edge("finalize", END)

    return builder.compile(store=store, checkpointer=checkpointer)

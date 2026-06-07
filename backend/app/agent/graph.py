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
    PLANNER_FEEDBACK_TEMPLATE,
    PLANNER_SYSTEM_PROMPT,
)
from app.agent.routing import executor_route, planner_route, route_after_eval
from app.agent.state import AgentState
from app.config import settings
from app.llm import get_chat_model

# グラフのノード名。グラフ構築とルーター（SSE イベントのフィルタリング）で共有し、
# リネーム時の文字列リテラルの取り残しを防ぐ。
NODE_RECALL = "recall"
NODE_PLANNER = "planner_agent"
NODE_PLANNER_TOOLS = "planner_tools"
NODE_EXECUTOR = "executor_agent"
NODE_EXECUTOR_TOOLS = "executor_tools"
NODE_EVALUATOR = "evaluator"
NODE_FINALIZE = "finalize"


def compute_recursion_limit() -> int:
    """グラフ実行に渡す recursion_limit を設定値から導出する。

    Planner→Executor→Evaluator のループ × ReAct ツール往復でステップ数が増えるため、
    試行ごとに planner/executor が各 (ツール往復×2 + 最終応答) ステップ + evaluator、
    さらに recall / finalize 分の余裕を持たせる。グラフ構造に依存する式のため、
    トポロジーを定義するこのモジュールに置く。
    """
    return settings.MAX_PLAN_ITERATIONS * (4 * settings.MAX_TOOL_TURNS + 4) + 4


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
    llm_plain = get_chat_model()
    llm_with_tools = llm_plain.bind_tools(tools)

    def _pick_model(scratch):
        """ツール往復が上限に達したら、ツールなしモデルで強制的に最終出力させる。

        ローカルモデルがツールを呼び続ける暴走の安全弁。AIMessage の数 = これまでの
        ツール往復回数。
        """
        tool_turns = sum(1 for m in scratch if isinstance(m, AIMessage))
        return llm_with_tools if tool_turns < settings.MAX_TOOL_TURNS else llm_plain

    async def _react_step(state: AgentState, scratch_key: str, sys_text: str, finalize):
        """Planner / Executor 共通の ReAct 1ステップ。

        ツール呼び出しが要求されたら scratch へ追記して戻し、最終出力なら
        finalize(response) の結果に scratch のリセットを添えて返す。
        """
        scratch = state.get(scratch_key) or []
        prompt = [SystemMessage(content=sys_text)] + state["messages"] + list(scratch)
        response = await _pick_model(scratch).ainvoke(prompt)
        if getattr(response, "tool_calls", None):
            return {scratch_key: [response]}
        return {**finalize(response), scratch_key: [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}

    async def planner_agent(state: AgentState) -> dict:
        """ユーザー要求と記憶から回答計画を立てるノード（scratch 上で ReAct）。"""
        sys_text = _build_system_text(
            PLANNER_SYSTEM_PROMPT,
            state.get("recalled_profile", ""),
            state.get("recalled_memories", ""),
        )
        feedback = state.get("feedback", "")
        if feedback:
            sys_text += PLANNER_FEEDBACK_TEMPLATE.format(feedback=feedback)
        return await _react_step(
            state,
            "planner_scratch",
            sys_text,
            lambda r: {
                "plan": _message_text(r),
                "iteration": state.get("iteration", 0) + 1,
            },
        )

    async def executor_agent(state: AgentState) -> dict:
        """計画に従ってユーザーへの最終回答ドラフトを生成するノード（scratch 上で ReAct）。"""
        sys_text = _build_system_text(
            EXECUTOR_SYSTEM_PROMPT,
            state.get("recalled_profile", ""),
            state.get("recalled_memories", ""),
        )
        sys_text += f"\n\n<plan>\n{state.get('plan', '')}\n</plan>"
        return await _react_step(
            state,
            "executor_scratch",
            sys_text,
            lambda r: {"draft": _message_text(r)},
        )

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
    builder.add_node(NODE_RECALL, recall_node)
    builder.add_node(NODE_PLANNER, planner_agent)
    builder.add_node(NODE_PLANNER_TOOLS, ToolNode(tools, messages_key="planner_scratch"))
    builder.add_node(NODE_EXECUTOR, executor_agent)
    builder.add_node(NODE_EXECUTOR_TOOLS, ToolNode(tools, messages_key="executor_scratch"))
    builder.add_node(NODE_EVALUATOR, evaluator_node)
    builder.add_node(NODE_FINALIZE, finalize_node)

    builder.add_edge(START, NODE_RECALL)
    builder.add_edge(NODE_RECALL, NODE_PLANNER)
    # Planner の ReAct ループ: ツール呼び出しがあれば実行して戻る、なければ Executor へ
    builder.add_conditional_edges(
        NODE_PLANNER,
        planner_route,
        {"tools": NODE_PLANNER_TOOLS, "next": NODE_EXECUTOR},
    )
    builder.add_edge(NODE_PLANNER_TOOLS, NODE_PLANNER)
    # Executor の ReAct ループ: 同上。完了したら Evaluator へ
    builder.add_conditional_edges(
        NODE_EXECUTOR,
        executor_route,
        {"tools": NODE_EXECUTOR_TOOLS, "next": NODE_EVALUATOR},
    )
    builder.add_edge(NODE_EXECUTOR_TOOLS, NODE_EXECUTOR)
    # 評価結果で分岐: 合格/上限到達 → finalize、不合格 → 再計画
    builder.add_conditional_edges(
        NODE_EVALUATOR,
        route_after_eval,
        {"finalize": NODE_FINALIZE, "planner_agent": NODE_PLANNER},
    )
    builder.add_edge(NODE_FINALIZE, END)

    return builder.compile(store=store, checkpointer=checkpointer)

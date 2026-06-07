from app.config import settings


def route_after_eval(state, max_iterations: int | None = None) -> str:
    """Evaluator の合否と試行回数から次ノードを決める条件エッジ。

    - ok → finalize（ドラフトを最終回答として確定）
    - ng & 上限到達 → finalize（無限ループ防止: 最後のドラフトを最良として採用）
    - ng & 上限未満 → planner_agent（フィードバック付きで再計画）
    """
    limit = max_iterations if max_iterations is not None else settings.MAX_PLAN_ITERATIONS
    verdict = (state.get("evaluation") or {}).get("verdict")
    if verdict == "ok":
        return "finalize"
    if state.get("iteration", 0) >= limit:
        return "finalize"
    return "planner_agent"


def _scratch_route(state, key: str) -> str:
    """scratch の末尾メッセージにツール呼び出しがあれば tools、なければ next。

    tools_condition は空の messages で例外を送出するため、scratch（計画確定時に
    空へリセットされる）に対しては使わず独自に判定する。
    """
    scratch = state.get(key) if isinstance(state, dict) else getattr(state, key, None)
    last = scratch[-1] if scratch else None
    if last is not None and getattr(last, "tool_calls", None):
        return "tools"
    return "next"


def planner_route(state) -> str:
    """Planner の ReAct 分岐: ツール実行か Executor への遷移かを決める。"""
    return _scratch_route(state, "planner_scratch")


def executor_route(state) -> str:
    """Executor の ReAct 分岐: ツール実行か Evaluator への遷移かを決める。"""
    return _scratch_route(state, "executor_scratch")

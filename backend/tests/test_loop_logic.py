"""Planner→Executor→Evaluator ループの純ロジック単体テスト（DB・Ollama 不要）。"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent import (
    executor_route,
    finalize_node,
    parse_evaluation,
    planner_route,
    route_after_eval,
)

# ---------------------------------------------------------------------------
# parse_evaluation: ローカルモデルの揺らぐ出力から合否を堅牢に抽出する
# ---------------------------------------------------------------------------


def test_parse_evaluation_valid_json():
    out = parse_evaluation('{"verdict": "ok", "score": 90, "feedback": ""}')
    assert out == {"verdict": "ok", "score": 90, "feedback": ""}


def test_parse_evaluation_ng_with_feedback():
    out = parse_evaluation('{"verdict": "ng", "score": 40, "feedback": "結論が曖昧"}')
    assert out["verdict"] == "ng"
    assert out["score"] == 40
    assert out["feedback"] == "結論が曖昧"


def test_parse_evaluation_strips_code_fences():
    text = '```json\n{"verdict": "ok", "score": 85, "feedback": ""}\n```'
    assert parse_evaluation(text)["verdict"] == "ok"


def test_parse_evaluation_with_surrounding_prose():
    text = '評価結果は以下のとおりです。\n{"verdict": "ng", "score": 30, "feedback": "計画に沿っていない"}\n以上です。'
    out = parse_evaluation(text)
    assert out["verdict"] == "ng"
    assert out["feedback"] == "計画に沿っていない"


def test_parse_evaluation_normalizes_verdict_aliases():
    assert parse_evaluation('{"verdict": "合格", "score": 80}')["verdict"] == "ok"
    assert parse_evaluation('{"verdict": "PASS", "score": 80}')["verdict"] == "ok"
    assert parse_evaluation('{"verdict": "不合格", "feedback": "短すぎる"}')["verdict"] == "ng"


def test_parse_evaluation_non_numeric_score_becomes_none():
    out = parse_evaluation('{"verdict": "ok", "score": "高い", "feedback": ""}')
    assert out["score"] is None


def test_parse_evaluation_broken_json_falls_back_to_keywords():
    # JSON が壊れていても本文のキーワードで判定する
    assert parse_evaluation("{verdict: 合格です（JSONが壊れている）")["verdict"] == "ok"
    out = parse_evaluation("不合格です。修正が必要: 結論を先に書くこと")
    assert out["verdict"] == "ng"
    assert "結論" in out["feedback"]


def test_parse_evaluation_fugoukaku_beats_goukaku():
    # 「不合格」は「合格」を部分文字列として含むため、NG 側を先に判定する
    assert parse_evaluation("残念ながら不合格です")["verdict"] == "ng"


def test_parse_evaluation_unparseable_defaults_to_ng():
    # 完全に判定不能なら ng（リトライ上限が安全弁になる）。生テキストを feedback に残す
    out = parse_evaluation("（意味不明な出力）")
    assert out["verdict"] == "ng"
    assert out["feedback"] == "（意味不明な出力）"


# ---------------------------------------------------------------------------
# route_after_eval: 合否と試行回数によるループ分岐
# ---------------------------------------------------------------------------


def test_route_after_eval_ok_goes_to_finalize():
    state = {"evaluation": {"verdict": "ok"}, "iteration": 1}
    assert route_after_eval(state, max_iterations=3) == "finalize"


def test_route_after_eval_ng_below_limit_retries():
    state = {"evaluation": {"verdict": "ng", "feedback": "改善して"}, "iteration": 1}
    assert route_after_eval(state, max_iterations=3) == "planner_agent"


def test_route_after_eval_ng_at_limit_finalizes():
    # 無限ループ防止: 上限到達時は最後のドラフトを最終回答として採用する
    state = {"evaluation": {"verdict": "ng", "feedback": "まだ駄目"}, "iteration": 3}
    assert route_after_eval(state, max_iterations=3) == "finalize"


def test_route_after_eval_missing_evaluation_retries_until_limit():
    assert route_after_eval({"iteration": 1}, max_iterations=3) == "planner_agent"
    assert route_after_eval({"iteration": 3}, max_iterations=3) == "finalize"


def test_route_after_eval_uses_settings_default():
    # max_iterations 省略時は settings.MAX_PLAN_ITERATIONS を使う
    from app.config import settings

    state = {
        "evaluation": {"verdict": "ng"},
        "iteration": settings.MAX_PLAN_ITERATIONS,
    }
    assert route_after_eval(state) == "finalize"


# ---------------------------------------------------------------------------
# planner_route / executor_route: scratch 上の ReAct 分岐
# ---------------------------------------------------------------------------


def _ai_with_tool_call():
    return AIMessage(
        content="",
        tool_calls=[{"name": "search_memory", "args": {"query": "好み"}, "id": "c1"}],
    )


def test_planner_route_with_tool_calls_goes_to_tools():
    state = {"planner_scratch": [_ai_with_tool_call()]}
    assert planner_route(state) == "tools"


def test_planner_route_empty_scratch_goes_next():
    # 計画確定時に scratch は空にリセットされる → 例外を出さず next へ
    assert planner_route({"planner_scratch": []}) == "next"
    assert planner_route({}) == "next"


def test_executor_route_after_tool_result_goes_next_when_no_more_calls():
    state = {
        "executor_scratch": [
            _ai_with_tool_call(),
            ToolMessage(content="[]", tool_call_id="c1"),
        ]
    }
    # 末尾が ToolMessage（ツール結果）の場合は LLM へ戻った後の判定なので next
    assert executor_route(state) == "next"


def test_executor_route_with_tool_calls_goes_to_tools():
    state = {"executor_scratch": [_ai_with_tool_call()]}
    assert executor_route(state) == "tools"


# ---------------------------------------------------------------------------
# finalize_node: 最終回答のみを会話履歴に確定する
# ---------------------------------------------------------------------------


def test_finalize_appends_draft_as_single_ai_message():
    state = {
        "messages": [HumanMessage("こんにちは")],
        "draft": "こんにちは！今日はどんなご用件ですか？",
        "plan": "- 挨拶を返す",
        "evaluation": {"verdict": "ok"},
    }
    out = finalize_node(state)

    assert list(out.keys()) == ["messages"]
    assert len(out["messages"]) == 1
    msg = out["messages"][0]
    assert isinstance(msg, AIMessage)
    assert msg.content == "こんにちは！今日はどんなご用件ですか？"


def test_finalize_with_empty_draft_uses_fallback_message():
    # ドラフトが空でも空バブルをユーザーに見せず、フォールバック文言を返す
    content = finalize_node({})["messages"][0].content
    assert content != ""
    assert "申し訳ありません" in content

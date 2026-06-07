import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk

from app.config import settings
from app.schemas import ChatRequest

router = APIRouter(tags=["chat"])

# グラフノード名 → フロントへ通知するフェーズ名。ここに無いノード（recall /
# tools / finalize）はフェーズ通知しない。
_PHASE_NODES = {
    "planner_agent": "planner",
    "executor_agent": "executor",
    "evaluator": "evaluator",
}


def _sse(event: str, data: dict) -> str:
    """1 件の SSE イベントを整形する（event 行 + data 行 + 空行区切り）。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _chunk_text(msg_chunk: AIMessageChunk) -> str:
    """AIMessageChunk からテキスト部分を取り出す（マルチパート content にも対応）。"""
    content = msg_chunk.content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content or ""


@router.post("/chat")
async def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    """マルチエージェントの進捗と生成トークンを SSE で逐次配信するチャットエンドポイント。

    astream_events(v2) でグラフのイベントを購読する:
    - on_chain_start（planner/executor/evaluator）: フェーズ開始を "phase" で通知
    - on_chain_end（planner_agent）: 確定した計画を "plan" で通知
    - on_chain_end（evaluator）: 評価結果を "evaluation" で通知。不合格で再計画が
      走る場合は "retry" も通知
    - on_chat_model_stream: LLM が吐くトークン。ユーザー向け最終回答を生成する
      executor_agent のものだけを "token" で配信（Planner / Evaluator の内部出力は
      流さない）。content が空のツール引数生成中チャンクは除外する
    - on_chain_end（ルート）: parent_ids が空 = グラフ全体の終了イベント。完全な
      最終状態（finalize 済みの messages）を取り出し、応答後のバックグラウンド
      記憶抽出へ渡す
    """
    agent = request.app.state.agent
    reflection_general = request.app.state.reflection_general
    reflection_profile = request.app.state.reflection_profile

    config = {"configurable": {"thread_id": req.thread_id, "user_id": req.user_id}}
    # Planner→Executor→Evaluator のループ × ReAct ツール往復でステップ数が増える
    # ため、グラフ実行にのみ recursion_limit を引き上げて渡す。上限は設定値から
    # 導出する: 試行ごとに planner/executor が各 (ツール往復×2 + 最終応答) ステップ
    # + evaluator、さらに recall / finalize 分の余裕を持たせる。
    recursion_limit = settings.MAX_PLAN_ITERATIONS * (4 * settings.MAX_TOOL_TURNS + 4) + 4
    graph_config = {**config, "recursion_limit": recursion_limit}

    async def event_stream():
        final_messages = None
        # 現在の試行回数。グラフ側の state["iteration"] と次の不変条件で同期する:
        # 「1試行につき planner の plan 確定（iteration インクリメント）はちょうど
        # 1回」。plan 確定時に output["iteration"] で正値に同期し、評価 NG で
        # ループが planner へ戻ると確定した時点で先行インクリメントする。
        attempt = 1
        last_phase = None  # ReAct 再入による同一フェーズの重複通知を抑止する

        try:
            async for ev in agent.astream_events(
                {"messages": [{"role": "user", "content": req.message}]},
                graph_config,
                version="v2",
            ):
                kind = ev["event"]

                if kind == "on_chat_model_stream":
                    # ユーザー向け回答を生成する executor のトークンだけを流す
                    node = ev.get("metadata", {}).get("langgraph_node")
                    if node != "executor_agent":
                        continue
                    text = _chunk_text(ev["data"]["chunk"])
                    if text:
                        yield _sse("token", {"text": text})

                elif kind == "on_chain_start" and ev.get("name") in _PHASE_NODES:
                    phase = _PHASE_NODES[ev["name"]]
                    if (phase, attempt) != last_phase:
                        last_phase = (phase, attempt)
                        yield _sse("phase", {"node": phase, "iteration": attempt})

                elif kind == "on_chain_end":
                    name = ev.get("name")
                    output = ev["data"].get("output")

                    if name == "planner_agent" and isinstance(output, dict):
                        # ReAct 途中（ツール呼び出し）の終了では plan を含まない
                        plan = output.get("plan")
                        if plan:
                            attempt = output.get("iteration", attempt)
                            yield _sse("plan", {"text": plan, "iteration": attempt})

                    elif name == "evaluator" and isinstance(output, dict):
                        evaluation = output.get("evaluation") or {}
                        if evaluation:
                            yield _sse(
                                "evaluation",
                                {
                                    "verdict": evaluation.get("verdict", "ng"),
                                    "score": evaluation.get("score"),
                                    "feedback": evaluation.get("feedback", ""),
                                    "iteration": attempt,
                                },
                            )
                            # 不合格かつ上限未満 → グラフは Planner へ戻る（再計画）
                            if (
                                evaluation.get("verdict") != "ok"
                                and attempt < settings.MAX_PLAN_ITERATIONS
                            ):
                                attempt += 1
                                yield _sse(
                                    "retry",
                                    {
                                        "iteration": attempt,
                                        "reason": evaluation.get("feedback", ""),
                                    },
                                )

                    elif not ev.get("parent_ids"):
                        # 親を持たない = グラフ全体のルート終了イベント。
                        # 累積された全 messages を含む最終状態を取り出す。
                        if isinstance(output, dict):
                            final_messages = output.get("messages")

            # ハイブリッド: 応答後にバックグラウンドで会話から記憶を抽出する。
            # finalize により messages は「ユーザー発話 + 最終回答」のみで、
            # ループ中の計画・評価は含まれない。
            if final_messages:
                payload = {"messages": final_messages}
                reflection_general.submit(payload, config=config, after_seconds=2)
                reflection_profile.submit(payload, config=config, after_seconds=2)
            yield _sse("done", {"thread_id": req.thread_id})
        except Exception as e:  # 生成途中の失敗をクライアントへ通知
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")

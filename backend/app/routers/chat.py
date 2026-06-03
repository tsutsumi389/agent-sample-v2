import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk

from app.schemas import ChatRequest

router = APIRouter(tags=["chat"])


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
    """LLM の生成トークンを SSE で逐次配信するチャットエンドポイント。

    astream_events(v2) でグラフのイベントを購読する:
    - on_chat_model_stream: LLM が吐くトークン。content が空のツール引数生成中
      チャンクは除外し、最終応答テキストのみ配信する。
    - on_chain_end(ルート): parent_ids が空 = グラフ全体の終了イベント。完全な
      最終状態（全 messages）を取り出し、応答後のバックグラウンド記憶抽出へ渡す。
    """
    agent = request.app.state.agent
    reflection = request.app.state.reflection

    config = {"configurable": {"thread_id": req.thread_id, "user_id": req.user_id}}

    async def event_stream():
        final_messages = None
        try:
            async for ev in agent.astream_events(
                {"messages": [{"role": "user", "content": req.message}]},
                config,
                version="v2",
            ):
                kind = ev["event"]

                if kind == "on_chat_model_stream":
                    # LLM のトークン1個（AIMessageChunk）
                    text = _chunk_text(ev["data"]["chunk"])
                    if text:
                        yield _sse("token", {"text": text})

                elif kind == "on_chain_end" and not ev.get("parent_ids"):
                    # 親を持たない = グラフ全体のルート終了イベント。
                    # 累積された全 messages を含む最終状態を取り出す。
                    output = ev["data"].get("output")
                    if isinstance(output, dict):
                        final_messages = output.get("messages")

            # ハイブリッド: 応答後にバックグラウンドで会話から記憶を抽出する
            if final_messages:
                reflection.submit(
                    {"messages": final_messages},
                    config=config,
                    after_seconds=2,
                )
            yield _sse("done", {"thread_id": req.thread_id})
        except Exception as e:  # 生成途中の失敗をクライアントへ通知
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")

from fastapi import APIRouter, Request

from app.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


def _extract_reply(result: dict) -> str:
    """エージェント応答メッセージから最後の AI テキストを取り出す。"""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        # langchain メッセージ or dict 両対応
        msg_type = getattr(msg, "type", None) or (
            msg.get("role") if isinstance(msg, dict) else None
        )
        if msg_type in ("ai", "assistant"):
            content = getattr(msg, "content", None)
            if content is None and isinstance(msg, dict):
                content = msg.get("content")
            if isinstance(content, list):
                # マルチパート content を結合
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            if content:
                return content
    return ""


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    agent = request.app.state.agent
    reflection = request.app.state.reflection

    config = {"configurable": {"thread_id": req.thread_id, "user_id": req.user_id}}

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": req.message}]},
        config,
    )

    reply = _extract_reply(result)

    # ハイブリッド: 応答後にバックグラウンドで会話から記憶を抽出（少し遅延させてデバウンス）
    reflection.submit(
        {"messages": result["messages"]},
        config=config,
        after_seconds=2,
    )

    return ChatResponse(reply=reply, thread_id=req.thread_id)

from fastapi import APIRouter, Query, Request

from app.schemas import MemoryItem

router = APIRouter(tags=["memories"])


def _iso(value) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


@router.get("/memories", response_model=list[MemoryItem])
async def list_memories(
    request: Request,
    user_id: str = Query(..., description="記憶を取得するユーザー"),
) -> list[MemoryItem]:
    """指定ユーザーの長期記憶を一覧取得する（記憶ビューア用）。"""
    store = request.app.state.store
    items = await store.asearch(("memories", user_id), limit=50)
    return [
        MemoryItem(
            key=item.key,
            value=item.value,
            created_at=_iso(getattr(item, "created_at", None)),
            updated_at=_iso(getattr(item, "updated_at", None)),
        )
        for item in items
    ]


@router.delete("/memories")
async def delete_memory(
    request: Request,
    user_id: str = Query(...),
    key: str = Query(..., description="削除する記憶のキー"),
) -> dict:
    """指定した記憶を削除する（デモのリセット用）。"""
    store = request.app.state.store
    await store.adelete(("memories", user_id), key)
    return {"deleted": key}

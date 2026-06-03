from fastapi import APIRouter, Query, Request

from app.schemas import MemoryItem

router = APIRouter(tags=["memories"])


def _iso(value) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _unwrap(value):
    """langmem の {"kind", "content"} エンベロープを展開して表示しやすい値にする。

    langmem は store に {"kind": <SchemaName>, "content": <本体>} 形式で保存する。
    フロント表示を安定させるため、エンベロープなら content 部分を返す。
    """
    if isinstance(value, dict) and set(value.keys()) == {"kind", "content"}:
        return value["content"]
    return value


def _to_item(item) -> MemoryItem:
    return MemoryItem(
        key=item.key,
        value=_unwrap(item.value),
        created_at=_iso(getattr(item, "created_at", None)),
        updated_at=_iso(getattr(item, "updated_at", None)),
    )


@router.get("/memories", response_model=list[MemoryItem])
async def list_memories(
    request: Request,
    user_id: str = Query(..., description="記憶を取得するユーザー"),
) -> list[MemoryItem]:
    """指定ユーザーの長期記憶（エピソード）を一覧取得する（記憶ビューア用）。"""
    store = request.app.state.store
    items = await store.asearch(("memories", user_id), limit=50)
    return [_to_item(item) for item in items]


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


@router.get("/profile", response_model=list[MemoryItem])
async def get_profile(
    request: Request,
    user_id: str = Query(..., description="プロフィールを取得するユーザー"),
) -> list[MemoryItem]:
    """指定ユーザーの構造化プロフィール（UserProfile）を取得する。"""
    store = request.app.state.store
    items = await store.asearch(("profile", user_id), limit=10)
    return [_to_item(item) for item in items]


@router.delete("/profile")
async def delete_profile(
    request: Request,
    user_id: str = Query(...),
    key: str = Query(..., description="削除するプロフィールのキー"),
) -> dict:
    """指定したプロフィールを削除する（デモのリセット用）。"""
    store = request.app.state.store
    await store.adelete(("profile", user_id), key)
    return {"deleted": key}

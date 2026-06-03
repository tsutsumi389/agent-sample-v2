from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore

from app.agent import build_agent
from app.config import settings
from app.llm import get_embeddings
from app.memory import build_general_reflection, build_profile_reflection
from app.routers import chat, memories
from app.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """起動時に Postgres ベースの store / checkpointer を初期化し、
    エージェントとバックグラウンド記憶抽出器を組み立てる。

    from_conn_string は非同期コンテキストマネージャを返すため、
    AsyncExitStack でアプリ生存中保持する。
    """
    async with AsyncExitStack() as stack:
        store = await stack.enter_async_context(
            AsyncPostgresStore.from_conn_string(
                settings.DATABASE_URL,
                index={"dims": settings.EMBED_DIMS, "embed": get_embeddings()},
            )
        )
        saver = await stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(settings.DATABASE_URL)
        )

        # 初回起動時に pgvector 拡張・テーブルを作成
        await store.setup()
        await saver.setup()

        app.state.store = store
        app.state.agent = build_agent(store, saver)
        # ハイブリッド記憶: 一般記憶（エピソード）と構造化プロフィールを別々に抽出する
        app.state.reflection_general = build_general_reflection(store)
        app.state.reflection_profile = build_profile_reflection(store)

        yield


app = FastAPI(title="Long-term Memory Agent Sample", lifespan=lifespan)

# 開発用 CORS（フロントは vite プロキシ経由が基本だが直接アクセスも許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")
app.include_router(memories.router, prefix="/api")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """DB と Ollama への疎通を確認する。"""
    db_status = "ok"
    try:
        await app.state.store.aget(("__healthcheck__",), "ping")
    except Exception as exc:  # noqa: BLE001
        db_status = f"error: {exc}"

    ollama_status = "ok"
    try:
        # 埋め込みを1回叩いて Ollama 到達性と次元を確認
        vec = await get_embeddings().aembed_query("ping")
        if len(vec) != settings.EMBED_DIMS:
            ollama_status = f"warn: dims={len(vec)} != {settings.EMBED_DIMS}"
    except Exception as exc:  # noqa: BLE001
        ollama_status = f"error: {exc}"

    overall = "ok" if db_status == "ok" and ollama_status == "ok" else "degraded"
    return HealthResponse(status=overall, database=db_status, ollama=ollama_status)

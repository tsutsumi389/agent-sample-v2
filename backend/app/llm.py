from functools import lru_cache

from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.config import settings


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOllama:
    """会話・記憶抽出に使う Ollama チャットモデル（プロセス内で単一インスタンスを共有）。"""
    return ChatOllama(
        model=settings.OLLAMA_CHAT_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.OLLAMA_TEMPERATURE,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    """長期記憶の意味検索に使う Ollama 埋め込みモデル（nomic-embed-text, 768次元）。

    /health がポーリングごとに呼ぶため、クライアントを毎回生成せず共有する。
    """
    return OllamaEmbeddings(
        model=settings.OLLAMA_EMBED_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )

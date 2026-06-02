from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.config import settings


def get_chat_model() -> ChatOllama:
    """会話・記憶抽出に使う Ollama チャットモデル。"""
    return ChatOllama(
        model=settings.OLLAMA_CHAT_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.3,
    )


def get_embeddings() -> OllamaEmbeddings:
    """長期記憶の意味検索に使う Ollama 埋め込みモデル（nomic-embed-text, 768次元）。"""
    return OllamaEmbeddings(
        model=settings.OLLAMA_EMBED_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )

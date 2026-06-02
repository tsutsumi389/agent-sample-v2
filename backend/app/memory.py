from langmem import ReflectionExecutor, create_memory_store_manager

from app.llm import get_chat_model

# 会話から抽出する記憶の指示。ハイブリッド構成の「バックグラウンド自動抽出」側。
MEMORY_INSTRUCTIONS = """会話からユーザーに関する長期的に有用な事実を抽出してください。
名前・好み・所属・興味・繰り返し言及される話題などを、簡潔な日本語の事実として記録します。
一時的・些末な情報は記録しないでください。"""


def build_reflection(store) -> ReflectionExecutor:
    """会話完了後にバックグラウンドで記憶を抽出・統合する実行器を構築する。

    namespace の "{user_id}" は submit 時の config.configurable.user_id から解決される。
    """
    manager = create_memory_store_manager(
        get_chat_model(),
        namespace=("memories", "{user_id}"),
        instructions=MEMORY_INSTRUCTIONS,
    )
    # store は ReflectionExecutor 経由でバックグラウンド実行時の config に注入される
    return ReflectionExecutor(manager, store=store)

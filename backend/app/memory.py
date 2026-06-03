from langmem import ReflectionExecutor, create_memory_store_manager

from app.llm import get_chat_model
from app.profile import PROFILE_INSTRUCTIONS, UserProfile

# 会話から抽出する一般記憶（エピソード）の指示。ハイブリッド構成の「随時追加」側。
MEMORY_INSTRUCTIONS = """会話からユーザーに関する長期的に有用な事実を抽出してください。
名前・好み・所属・興味・繰り返し言及される話題などを、簡潔な日本語の事実として記録します。
一時的・些末な情報は記録しないでください。"""


def build_general_reflection(store) -> ReflectionExecutor:
    """会話完了後にフリーテキストの一般記憶を抽出・統合する実行器を構築する。

    namespace の "{user_id}" は submit 時の config.configurable.user_id から解決される。
    """
    manager = create_memory_store_manager(
        get_chat_model(),
        namespace=("memories", "{user_id}"),
        instructions=MEMORY_INSTRUCTIONS,
    )
    return ReflectionExecutor(manager, store=store)


def build_profile_reflection(store) -> ReflectionExecutor:
    """会話完了後に構造化プロフィール（UserProfile）を抽出・更新する実行器を構築する。

    ("profile", "{user_id}") の単一プロフィールを upsert し続ける（enable_inserts=True で
    初回作成、以降は既存プロフィールを更新）。schemas 指定により構造化抽出になる。
    """
    manager = create_memory_store_manager(
        get_chat_model(),
        schemas=[UserProfile],
        namespace=("profile", "{user_id}"),
        instructions=PROFILE_INSTRUCTIONS,
        enable_inserts=True,
        enable_deletes=False,
    )
    return ReflectionExecutor(manager, store=store)

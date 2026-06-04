from langmem import ReflectionExecutor, create_memory_store_manager

from app.llm import get_chat_model
from app.profile import PROFILE_INSTRUCTIONS, UserProfile

# 会話から抽出する一般記憶（エピソード）の指示。ハイブリッド構成の「随時追加」側。
# 安定したプロフィール情報は build_profile_reflection（UserProfile スキーマ）側の責務の
# ため、ここではエピソード的な事実に限定して棲み分ける（二重保存・二重注入の防止）。
MEMORY_INSTRUCTIONS = """会話からユーザーに関する長期的に有用な事実を抽出してください。
出来事・予定・進行中の話題・相談内容の経緯など、エピソード的な事実が対象です。
名前・属性・好み・応答スタイルなどの安定したプロフィール情報は別の仕組みで抽出されるため、
ここでは記録しないでください。
1件につき1つの事実を、簡潔な日本語で記録してください。
相対的な日時表現（来週・先月など）は、会話から特定できる場合は具体的な時期に言い換えてください。
ユーザー自身の発言に基づく事実のみを記録し、アシスタントの発言は事実として記録しないでください。
一時的・些末な情報は記録しないでください。
機微情報（健康状態・政治信条・宗教など）はユーザーが明言した場合のみ記録してください。"""


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

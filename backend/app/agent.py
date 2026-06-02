from langgraph.prebuilt import create_react_agent
from langmem import create_manage_memory_tool, create_search_memory_tool

from app.llm import get_chat_model

# エージェントへのシステム指示。記憶ツールの能動的な利用を促す（gemma4 のような
# ローカルモデルでもツール呼び出しを行いやすくするため明示する）。
SYSTEM_PROMPT = """あなたは長期記憶を持つ親切な日本語アシスタントです。

記憶の扱い方:
- ユーザーに関する新しい事実（名前・好み・所属・予定など）を知ったら、
  manage_memory ツールで保存してください。
- ユーザーについて尋ねられたら、まず search_memory ツールで過去の記憶を検索し、
  見つかった情報をもとに答えてください。
- 記憶が見つからない場合は、知らない旨を正直に伝えてください。
"""


def build_agent(store, checkpointer):
    """ホットパス型の記憶ツールを備えた ReAct エージェントを構築する。

    - store: 長期記憶（pgvector 意味検索付き）
    - checkpointer: 会話履歴（thread_id 単位の短期記憶）

    namespace の "{user_id}" は ainvoke 時の config.configurable.user_id から
    実行時に解決される。
    """
    return create_react_agent(
        get_chat_model(),
        prompt=SYSTEM_PROMPT,
        tools=[
            create_manage_memory_tool(namespace=("memories", "{user_id}")),
            create_search_memory_tool(namespace=("memories", "{user_id}")),
        ],
        store=store,
        checkpointer=checkpointer,
    )

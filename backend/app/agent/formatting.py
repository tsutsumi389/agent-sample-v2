import json


def _format_items(items) -> str:
    """store の検索結果アイテム列を、1行1件の JSON（JSONL）に整形する。

    langmem の保存形式（{"kind", "content"} エンベロープ等）をそのまま出すことで、
    形状判定による剥がしロジックを持たない。日本語が \\uXXXX にエスケープされない
    よう ensure_ascii=False を指定する。
    """
    return "\n".join(
        json.dumps(item.value, ensure_ascii=False) for item in items if item.value
    )


def _build_system_text(base: str, recalled_profile: str, recalled_memories: str) -> str:
    """想起済みのプロフィール・記憶ブロックをシステムプロンプトへ合成する。

    プロフィール（常時注入の確定情報）と話題依存の記憶を XML タグで区切り、
    LLM にデータの境界と確度の違いを伝える。
    """
    sys_text = base
    if recalled_profile:
        sys_text += f"\n\n<user_profile>\n{recalled_profile}\n</user_profile>"
    if recalled_memories:
        sys_text += f"\n\n<related_memories>\n{recalled_memories}\n</related_memories>"
    return sys_text


def _message_text(message) -> str:
    """メッセージからテキスト部分を取り出す（マルチパート content にも対応）。"""
    content = message.content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content or ""

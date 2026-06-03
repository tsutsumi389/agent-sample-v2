from pydantic import BaseModel, Field

# 構造化プロフィールのスキーマ。フリーテキストの一般記憶（("memories", user_id)）とは
# 別の namespace（("profile", user_id)）に、単一の集約オブジェクトとして upsert される。
# パーソナライズの「安定した属性・嗜好・応答スタイル」をここに集約する。


class UserProfile(BaseModel):
    """ユーザーの安定した属性・嗜好・応答スタイル（会話を跨いで保持）。"""

    name: str | None = Field(None, description="ユーザーの名前・呼び名")
    locale: str | None = Field(None, description="使用言語・地域（例: ja-JP）")
    attributes: list[str] = Field(
        default_factory=list, description="所属・職業・立場などの安定した属性"
    )
    preferences: list[str] = Field(
        default_factory=list, description="好み・興味・嗜好（例: コーヒーが好き）"
    )
    response_style: str | None = Field(
        None, description="好む応答スタイル（口調・詳しさ・形式の指示。例: 簡潔に）"
    )


# プロフィール抽出用のバックグラウンド指示（ハイブリッド構成のプロフィール側）。
PROFILE_INSTRUCTIONS = """会話からユーザーの安定したプロフィールを抽出・更新してください。
名前・使用言語・所属や職業などの属性・好みや興味・好む応答スタイルを対象にします。
既存のプロフィールがあれば上書き更新し、一時的・些末な情報は含めないでください。"""

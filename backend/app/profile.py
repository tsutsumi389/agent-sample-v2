from pydantic import BaseModel, Field

# 構造化プロフィールのスキーマ。フリーテキストの一般記憶（("memories", user_id)）とは
# 別の namespace（("profile", user_id)）に、単一の集約オブジェクトとして upsert される。
# パーソナライズの「安定した属性・嗜好・応答スタイル」をここに集約する。
# 項目構成はトレンド調査レポート（docs/personalization-trends-research.html 第5.1章）の
# 推奨に基づく: アイデンティティ／属性／嗜好／コミュニケーション／文脈の5カテゴリ。


class UserProfile(BaseModel):
    """ユーザーの安定した属性・嗜好・応答スタイル（会話を跨いで保持）。"""

    # アイデンティティ
    name: str | None = Field(None, description="ユーザーの名前・呼び名")
    locale: str | None = Field(None, description="使用言語・地域（例: ja-JP）")
    timezone: str | None = Field(
        None, description="タイムゾーン（例: Asia/Tokyo。日時表現のパーソナライズに使用）"
    )
    # 属性
    attributes: list[str] = Field(
        default_factory=list, description="所属・職業・立場などの安定した属性"
    )
    expertise: list[str] = Field(
        default_factory=list,
        description="専門領域と習熟度（例: Python上級、機械学習は初学者）",
    )
    # 嗜好
    preferences: list[str] = Field(
        default_factory=list, description="好み・興味・嗜好（例: コーヒーが好き）"
    )
    dislikes: list[str] = Field(
        default_factory=list,
        description="苦手・避けたいもの（NG話題・苦手な形式など）",
    )
    # コミュニケーション
    response_style: str | None = Field(
        None, description="好む応答スタイル（口調・詳しさ・形式の指示。例: 簡潔に）"
    )
    # 文脈
    goals: list[str] = Field(
        default_factory=list,
        description="中長期の目標・進行中のプロジェクト（例: 資格試験勉強中）",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="制約・配慮事項（アレルギー・利用環境・予算・時間制約など）",
    )


# プロフィール抽出用のバックグラウンド指示（ハイブリッド構成のプロフィール側）。
# 抽出対象の項目はスキーマ（UserProfile の各 Field description）が LLM に渡るため
# ここでは列挙せず、スキーマで表現できないポリシーのみを記述する。
PROFILE_INSTRUCTIONS = """会話からユーザーの安定したプロフィールを抽出・更新してください。
抽出対象はスキーマの各フィールド定義に従ってください。
一時的・些末な情報は含めないでください。
機微情報（健康状態・政治信条・宗教など）はユーザーが明言した場合のみ記録し、推測で保存しないでください。"""

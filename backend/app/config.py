from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """環境変数から読み込むアプリ設定。"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # PostgreSQL（psycopg3 を使うため postgresql:// 形式）
    DATABASE_URL: str = "postgresql://app:app@localhost:5432/agentdb"

    # Ollama（ホストで起動済み）
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_CHAT_MODEL: str = "gemma4:latest"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text:latest"

    # nomic-embed-text の埋め込み次元。store の index.dims と一致させる。
    EMBED_DIMS: int = 768

    # Planner→Executor→Evaluator ループの最大試行回数。Evaluator が不合格を
    # 出し続けた場合の安全弁で、到達時は最後のドラフトを最終回答として採用する。
    MAX_PLAN_ITERATIONS: int = 3

    # Planner / Executor の ReAct ループにおけるツール往復回数の上限。
    # ローカルモデルがツールを呼び続ける暴走の安全弁。
    MAX_TOOL_TURNS: int = 5


settings = Settings()

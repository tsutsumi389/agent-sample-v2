# agent-sample-v2 — 長期記憶を持つAIエージェント サンプル

会話を跨いでユーザーに関する情報を記憶し、後の会話で活用できる AI エージェントのサンプル実装です。

- **バックエンド**: Python / FastAPI / LangGraph / LangMem
- **フロントエンド**: React / TypeScript（Vite）
- **DB**: PostgreSQL + pgvector（意味検索）
- **LLM**: ローカル Ollama（チャット: `gemma4:latest`、埋め込み: `nomic-embed-text:latest`）

## アーキテクチャ

```
React (Chat + 記憶ビューア)  ──/api──▶  FastAPI
                                          └ LangGraph マルチエージェント
                                              ├ Planner（計画立案） ⇄ 記憶ツール
                                              ├ Executor（回答生成） ⇄ 記憶ツール
                                              ├ Evaluator（品質評価）─ NG → Planner へ再計画
                                              ├ ReflectionExecutor（背景で記憶抽出）
                                              ├ Checkpointer = 会話履歴（thread_id 単位）
                                              └ Store = 長期記憶（user_id 単位 / pgvector）
                                          │
                            PostgreSQL(pgvector)        Ollama（ホスト）
```

### マルチエージェントループ（Planner → Executor → Evaluator）

```
START → recall（記憶想起）
      → Planner（回答計画を立案）
      → Executor（計画に従い回答を生成 = SSE でトークン配信）
      → Evaluator（回答を評価）
           ├ 合格 / 最大試行回数（MAX_PLAN_ITERATIONS, 既定3）到達 → 回答確定 → END
           └ 不合格 → フィードバック付きで Planner へ戻り再計画
```

- 計画・評価・リトライ状況は SSE イベント（`phase` / `plan` / `evaluation` / `retry`）で
  フロントエンドへ配信され、チャット UI の「エージェントの動き」に表示されます。
- 会話履歴（checkpointer）には最終回答のみが残り、計画・評価などのループ内部情報は
  履歴を汚しません。

### 記憶の仕組み（ハイブリッド）
1. **ホットパス型**: エージェントが会話中に `manage_memory` / `search_memory` ツールを明示的に呼び出して保存・検索。
2. **バックグラウンド型**: 応答後に `ReflectionExecutor` が会話から重要な事実を自動抽出して保存。

長期記憶は `namespace=("memories", <user_id>)` で、会話履歴は `thread_id` で分離されます。

## 前提

- Docker / Docker Compose
- ホストで Ollama が起動済みで、以下のモデルが pull 済みであること:
  ```bash
  ollama pull gemma4         # 実際のタグは gemma4:latest
  ollama pull nomic-embed-text
  ```

## 起動

```bash
cp .env.example .env
docker compose up --build
```

- フロントエンド: http://localhost:5173
- バックエンド API: http://localhost:8000 （`/docs` に Swagger UI）

> コンテナからホストの Ollama へは `host.docker.internal:11434` で接続します（`.env` の `OLLAMA_BASE_URL`）。

## 動作確認（CLI）

```bash
# 1. 疎通確認
curl localhost:8000/health

# 2. 記憶を保存（応答は SSE でトークンを逐次配信。-N でバッファリングを無効化）
curl -N -X POST localhost:8000/api/chat -H 'content-type: application/json' \
  -d '{"user_id":"alice","thread_id":"t1","message":"私の名前は田中で、コーヒーが好きです"}'

# 3. 数秒後、抽出された長期記憶を確認
curl "localhost:8000/api/memories?user_id=alice"

# 4. 別スレッドで想起（長期記憶が会話を跨ぐ）
curl -N -X POST localhost:8000/api/chat -H 'content-type: application/json' \
  -d '{"user_id":"alice","thread_id":"t2","message":"私の好きな飲み物は何でしたか？"}'
```

`docker compose restart backend` で再起動しても、記憶（DB）は保持されます。

## ディレクトリ

```
backend/   FastAPI + LangGraph + LangMem
frontend/  React + TypeScript (Vite)
docker-compose.yml
```

## ⚠️ セキュリティに関する注意（ローカルデモ専用・本番利用不可）

本リポジトリは **記憶の仕組みを学ぶためのローカルデモ** であり、認証を意図的に省いています。
そのため、以下の既知のリスクがあります。**インターネットや共有環境に公開しないでください。**

- **認証なし / なりすまし**: `user_id` をクライアント（リクエストボディ・クエリ）から受け取り、そのまま信頼します。任意の `user_id` を指定すれば他人の記憶を閲覧・削除でき、別ユーザーとして発話できます（IDOR）。
- **thread_id の検証なし**: 会話スレッドの所有者検証を行いません。
- **寛容な CORS**: `allow_origins=["*"]` のため、任意のオリジンからのブラウザアクセスを許可します。
- **レート制限なし**。

### 本番化する場合の対応指針

1. JWT / セッション等で認証し、`user_id` は **検証済みプリンシパルから導出**（リクエストの `user_id` は廃止）
2. `thread_id` をサーバ側で発番してユーザーに紐付け、各呼び出しで所有者を検証
3. `allow_origins` を既知のフロントオリジンに限定（ワイルドカード廃止）
4. `/api/chat`・`/api/memories` にレート制限を付与

## その他の注意点

- `nomic-embed-text` の埋め込み次元は **768**。`.env` の `EMBED_DIMS` と一致させてください。
- LangGraph の Postgres 実装は psycopg3 を使うため、接続文字列は `postgresql://`（`+psycopg2` は付けない）。
- ローカルモデル（gemma4）はツール呼び出しが不安定な場合があります。その際もバックグラウンド抽出が記憶の主経路として機能します。

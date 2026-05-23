# Urban Safety Official Agent Backend

愛知県の公式情報を取得し、LLMで構造化したうえで、回答生成 Agent と Verifier Agent により利用者への表示可否を制御する FastAPI バックエンドです。

## 起動

```bash
python -m pip install -r requirements.txt
python init_db.py
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## データフロー

```text
公式情報源
  ↓
official/sync 取得
  ↓
LLM またはルールで構造化
  ↓
公式情報 DB
  ↓
ユーザー質問
  ↓
回答生成 Agent
  ↓
draft_answer
  ↓
Verifier Agent
  ↓
PASS / FAIL / NEEDS_REVIEW
  ↓
SHOW / SHOW_WITH_WARNING / DO_NOT_SHOW
  ↓
UI 表示
```

## 主な API

- `GET /dashboard`: 保存済み公式情報の要約と一覧を返します。
- `POST /official/sync?force=true&limit=1`: 公式情報源へアクセスし、LLMで構造化してDBへ保存します。
- `GET /official/live`: DBに保存済みの公式情報を返します。
- `GET /official/sources`: 取得対象の公式情報カタログを返します。
- `GET /official/live/weather`: 気象庁防災情報XMLから愛知県関連情報を取得します。
- `POST /ask`: 公式情報DBを根拠に回答を生成し、Verifier Agent の判定結果と表示ポリシーを返します。
- `DELETE /answers/cache`: 保存済みの回答検証履歴を削除します。
- `GET /system/overview`: 現在のパイプラインとDB状態を返します。
- `GET /health`: DBとAI接続状態を確認します。

## AI 設定

未指定の場合はローカル Ollama の `qwen3.6:35b-a3b` を使います。必要な場合だけ環境変数で上書きしてください。
`AI_MODEL` は全体のデフォルトで、Generator / Verifier / 公式情報正規化は個別にも指定できます。

```bash
AI_PROVIDER=ollama
AI_MODEL=qwen3.6:35b-a3b
AI_GENERATOR_MODEL=qwen3.6:35b-a3b
AI_VERIFIER_MODEL=qwen3.6:35b-a3b
AI_NORMALIZER_MODEL=qwen3.6:35b-a3b
AI_BASE_URL=http://localhost:11434/api/generate
```

API互換サーバーを使う場合は `AI_PROVIDER=api`、`AI_BASE_URL`、各モデル名、必要に応じて `AI_API_KEY` を設定します。

## 定期取得

バックエンド起動中は、デフォルトで30分ごとに公式情報の取得可否を確認します。手動更新は `force=true` によりキャッシュ間隔を無視して取得します。

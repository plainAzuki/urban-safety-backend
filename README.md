# Urban Safety AI Agent Backend

愛知県内の都市安全情報を想定し、SNS風の模擬投稿、気象・鉄道・交通に相当する公式信号、LLMによる行動提案を統合する FastAPI バックエンドです。

## 目的

実SNSの収集が難しい前提で、模擬SNS投稿を住民の現場感知データとして扱います。そこに公式情報由来の信号を重ね、カテゴリ別の重み付けで総合リスクを算出します。

## 起動

```bash
python -m pip install -r requirements.txt
python init_db.py
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 主なAPI

- `GET /dashboard`: アプリ首页用のリスク一覧、集計、時間推移を返します。
- `POST /analyze/{event_id}`: LLMまたはルール型 fallback で詳細助言を返します。
- `POST /official/sync`: 模擬値から公式API相当の信号をDBへ保存します。
- `GET /official/sources`: 実APIへ差し替える対象の公式情報カタログを返します。
- `GET /official/live/weather`: 気象庁防災情報XMLのAtomフィードから愛知県関連の見出しを取得します。
- `POST /official/live/sync`: 取得した公式ライブ情報を地域単位の観測値としてDBへ保存します。
- `GET /official/live`: DBに保存済みの公式ライブ情報を返します。
- `GET /system/overview`: 卒業制作の説明に使えるデータフローと設定概要を返します。
- `GET /health`: DBとAI接続状態を確認します。

## GPT等のオンラインAPIへ切り替える場合

`.env.example` を参考に、以下を設定します。

```bash
export AI_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4.1-mini
```

データベース、リスクスコア、アプリ画面はそのまま使えます。差し替える中心は `main.py` の `call_openai` と環境変数です。

## 簡易確認

```bash
python smoke_test.py
python api_contract_test.py
```

Ollama が起動していない場合でも、詳細分析は `rule-based-fallback` により参考提案を返します。

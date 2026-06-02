# Urban Safety Research Backend

公的情報と模擬イベントデータに基づく都市安全情報集約・可視化システムの FastAPI バックエンドです。

本プロジェクトは、実運用可能な防災製品ではなく、卒業研究として「公的情報と模擬データを組み合わせた都市安全情報支援システムの設計・実装および可行性検証」を行うための原型システムです。

## 研究テーマ

公的情報と模擬イベントデータに基づく都市安全情報集約・可視化システムの設計と実装

## 起動

```bash
python -m pip install -r requirements.txt
python init_db.py
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## データフロー

```text
公的情報源
  ↓
official/sync で取得
  ↓
LLM またはルールで統一データモデルへ構造化
  ↓
都市安全情報DB
  ↑
研究用模擬イベントデータ
  ↓
API
  ↓
モバイルUIで通常時・異常時を可視化
  ↓
自然言語問い合わせに対して保存済み情報から要約回答
```

## 主な API

- `GET /dashboard?include_simulated=false`: 画面用の概要と最新情報を返します。
- `GET /safety/events`: 統一データモデルの都市安全情報一覧を返します。
- `POST /official/sync?force=true&limit=1`: 公的情報源へアクセスし、DBへ保存します。
- `GET /official/live`: DBに保存済みの情報を返します。
- `GET /official/sources`: 取得対象の公的情報カタログを返します。
- `GET /safety/simulated-events/scenarios`: 研究用の模擬シナリオ一覧を返します。
- `POST /safety/simulated-events/load?scenario=ollama_random&mode=replace&count=10&dangerous_ratio=0.3`: ローカル Ollama で模擬イベントを生成し、保存します。
- `DELETE /safety/simulated-events`: 模擬イベントだけを削除します。
- `POST /ask`: 保存済み情報に基づく要約回答、参照情報、模擬データ有無を返します。
- `GET /system/overview`: 現在の研究用パイプラインとDB状態を返します。
- `GET /health`: DBとAI接続状態を確認します。

## 模擬データの扱い

模擬データは公的情報ではありません。すべての模擬イベントには以下を付与します。

- `is_simulated: true`
- `source: 研究用Ollama模擬イベントデータ`
- `display_label` の「模擬データ」表示

模擬データはコード内の固定データではなく、ローカル Ollama に固定JSON形式で生成させます。標準設定では10件を生成し、低リスク・無危険を約7割、高リスク・支障ありを約3割にします。

## AI 設定

未指定の場合はローカル Ollama の `qwen3.6:35b-a3b` を使います。AI が利用できない場合でも、保存済み情報から決定的なフォールバック要約を返します。

```bash
AI_PROVIDER=ollama
AI_MODEL=qwen3.6:35b-a3b
AI_GENERATOR_MODEL=qwen3.6:35b-a3b
AI_NORMALIZER_MODEL=qwen3.6:35b-a3b
AI_BASE_URL=http://localhost:11434/api/generate
```


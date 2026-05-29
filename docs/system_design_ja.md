# システム設計

## システム構成

```text
公的情報源
  ↓
バックエンド取得処理
  ↓
統一データモデルへ構造化
  ↓
都市安全情報DB
  ↑
研究用模擬イベント投入
  ↓
API
  ↓
React Native / Expo フロントエンド
```

本システムは、公的情報の取得と研究用模擬イベントを同じ都市安全情報DBで扱う。ただし、模擬イベントには必ず `is_simulated: true` を付与し、APIとUIで「模擬データ」と表示する。

## バックエンド構成

- `main.py`: FastAPI のルーティング。
- `db.py`: SQLite のテーブル定義、保存、検索、模擬データ削除。
- `official_sources.py`: 公的情報源の定義と取得処理。
- `official_service.py`: 公的情報の正規化、保存、概要生成。
- `simulated_events.py`: ローカル Ollama による研究検証用の模擬イベント生成。
- `answer_service.py`: 保存済み都市安全情報に基づく要約回答。
- `schemas.py`: API リクエストモデル。
- `prompts.py`: LLM を使う場合の正規化・要約用プロンプト。

## フロントエンド構成

- `App.js`: 画面全体の状態管理、日常データと模擬データの切替。
- `src/api/urbanSafetyApi.js`: バックエンドAPI呼び出し。
- `src/components/SummaryPanel.js`: 都市安全情報の概要表示。
- `src/components/OfficialCard.js`: 情報カード表示。
- `src/components/OfficialInfoModal.js`: 詳細・履歴表示。
- `src/components/AskPanel.js`: 自然言語問い合わせと参照情報表示。
- `src/styles/styles.js`: 共通スタイル。
- `src/utils/official.js`: 状態ラベルと表示色。

## データモデル

都市安全情報は、最低限以下の項目を持つ。

| 項目 | 説明 |
|---|---|
| `source` | 情報源名 |
| `category` | 鉄道、道路、気象、防災など |
| `area` | 対象地域 |
| `label` | 情報タイトル |
| `display_label` | フロントエンド表示用ラベル |
| `detail` | 詳細説明 |
| `status` | 通常、情報、注意、警戒、危険、運休、支障、取得不可 |
| `severity` | 重要度・リスクレベル |
| `observed_at` | 発生・観測時刻 |
| `updated_at` | 更新時刻 |
| `source_url` | 出典URL |
| `is_simulated` | 模擬データかどうか |

## 公的情報と模擬データの扱い

公的情報は、気象庁、自治体防災情報、交通事業者、道路交通情報などから取得する。模擬データは研究検証のためにローカル Ollama が固定JSON形式で生成する。標準設定では20件を生成し、低リスク・無危険を約3割、高リスク・支障ありを約7割にする。模擬データは `source` に「研究用Ollama模擬イベントデータ」、`is_simulated` に `true` を設定する。

## API概要

| API | 目的 |
|---|---|
| `GET /dashboard?include_simulated=false` | 概要画面用の最新情報を取得 |
| `GET /official/live?include_simulated=false` | 保存済み情報の一覧・履歴を取得 |
| `GET /safety/events` | 統一データモデルの都市安全情報一覧を取得 |
| `POST /official/sync` | 公的情報源から情報を取得・保存 |
| `GET /safety/simulated-events/scenarios` | 利用可能な模擬シナリオ一覧を取得 |
| `POST /safety/simulated-events/load?scenario=ollama_random&mode=replace&count=20&dangerous_ratio=0.7` | Ollama生成の模擬イベントをDBへ投入 |
| `DELETE /safety/simulated-events` | 模擬イベントのみ削除 |
| `POST /ask` | 保存済み情報に基づく要約回答と参照情報を取得 |

## API呼び出し例

```bash
curl "http://localhost:8000/dashboard?include_simulated=true"
curl "http://localhost:8000/safety/events?include_simulated=true&category=鉄道"
curl -X POST "http://localhost:8000/safety/simulated-events/load?scenario=ollama_random&mode=replace&count=20&dangerous_ratio=0.7"
curl -X DELETE "http://localhost:8000/safety/simulated-events"
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"中部国際空港へ向かう時に注意すべき情報は？","include_simulated":true}'
```

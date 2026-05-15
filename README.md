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
  - `refresh=true` を付けると保存済み分析を使わず再生成します。
- `POST /official/sync`: 模擬値から公式API相当の信号をDBへ保存します。
- `GET /official/sources`: 実APIへ差し替える対象の公式情報カタログを返します。
- `GET /official/live/weather`: 気象庁防災情報XMLのAtomフィードから愛知県関連の見出しを取得します。
- `POST /official/live/sync`: 取得した公式ライブ情報を地域単位の観測値としてDBへ保存します。
- `GET /official/live`: DBに保存済みの公式ライブ情報を返します。
- `GET /system/overview`: 卒業制作の説明に使えるデータフローと設定概要を返します。
- `GET /evaluation/summary`: SNSのみ、多ソース融合ありなどの比較実験結果を返します。
- `GET /health`: DBとAI接続状態を確認します。
- `DELETE /analysis/cache`: 保存済みAI分析を削除します。`event_id` 指定も可能です。

## GPT等のオンラインAPIへ切り替える場合

`.env.example` を参考に、以下を設定します。

```bash
export AI_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4.1-mini
```

データベース、リスクスコア、アプリ画面はそのまま使えます。差し替える中心は `main.py` の `call_openai` と環境変数です。

## 公式情報ソース

気象情報は、気象庁防災情報XMLのAtomフィードを利用する入口を実装しています。

- PULL型公開ページ: `https://xml.kishou.go.jp/xmlpull.html`
- デフォルトfeed: `https://www.data.jma.go.jp/developer/xml/feed/extra_l.xml`
- 愛知県コード: `230000`

現在は電文タイトルと本文テキストを軽量に正規化し、`official_area_observations` に保存します。本文XMLの詳細構造をさらに解析すれば、市区町村別・警報種別別の精度を上げられます。

## 簡易確認

```bash
python evaluation.py
python smoke_test.py
python api_contract_test.py
```

Ollama が起動していない場合でも、詳細分析は `rule-based-fallback` により参考提案を返します。

## 現在のリスク表示ロジック

- `risk_score`: SNS投稿数、気象リスク、交通・鉄道リスクをカテゴリ別重みで統合します。
- `risk_factors`: スコア内訳を `sns / weather / transport` で返します。
- `confidence_score`: SNS投稿数と公式信号の有無から情報の確からしさを返します。
- `action_plan`: LLMに依存しない短い推奨行動を返します。
- `risk_timeline`: 過去時間帯ごとのリスク集中を返します。
- `ai_analyses`: 詳細分析を保存し、同一モデルでは再生成を避けます。

## 研究評価

`evaluation.py` は、報告書で指摘された「SNSのみ vs 多ソース融合あり」の比較を行うための評価スクリプトです。

- `evaluation_dataset.csv`: `post_id`, `event_id`, `is_noise`, `event_type`, `official_match`, `ground_truth_risk` などを固定した評価用データです。
- `evaluation_results.json`: A キーワード方式、B SNSのみ、C SNS+気象、D SNS+気象+交通の Precision / Recall / F1 を保存します。
- `evaluation_metrics.csv`: 表計算ソフトで棒グラフを作りやすい条件別指標です。
- `evaluation_failures.csv`: 条件別の誤検知・見逃し例を保存します。
- `evaluation_metrics_chart.svg`: Precision / Recall / F1 の棒グラフです。
- `evaluation_report.md`: 次回報告・卒論下書きに貼り付けやすい評価レポートです。
- LLMは検知精度に混ぜず、構造化データによるリスク判定だけを評価します。
- 予測時には正解ラベルの `is_noise` を使わず、投稿本文から推定した `detected_event_type` を使います。
- 公式信号の紐づけは、投稿前後3時間、半径5km以内、イベント種別の整合性という3条件で判定します。
- DBSCAN相当の時空間クラスタリングにより、Duplicate Reduction Rate と Cluster Purity も算出します。

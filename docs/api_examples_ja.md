# API仕様・呼び出し例

## 概要取得

```bash
curl "http://localhost:8000/dashboard?include_simulated=false"
curl "http://localhost:8000/dashboard?include_simulated=true"
```

## 統一都市安全情報一覧

```bash
curl "http://localhost:8000/safety/events?include_simulated=true"
curl "http://localhost:8000/safety/events?include_simulated=true&category=鉄道"
curl "http://localhost:8000/safety/events?include_simulated=true&area=常滑&min_severity=2"
```

## 模擬イベント

```bash
curl "http://localhost:8000/safety/simulated-events/scenarios"
curl -X POST "http://localhost:8000/safety/simulated-events/load?scenario=multi_event&mode=replace"
curl -X DELETE "http://localhost:8000/safety/simulated-events"
```

## 自然言語問い合わせ

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"今、愛知県で注意すべき情報は？","include_simulated":false}'
```

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"中部国際空港へ向かう時に注意すべき情報は？","include_simulated":true}'
```

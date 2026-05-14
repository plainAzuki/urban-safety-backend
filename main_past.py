"""
Urban Safety AI Agent - Backend
FastAPI + Ollama (Llama 3.2 11B)
"""

import json
import httpx
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── 設定 ────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
# ★ ollama list で確認したモデル名に合わせてください
OLLAMA_MODEL = "llama3.2-vision:latest"
EVENTS_FILE = Path(__file__).parent / "events.json"

# カテゴリ別の重み設定（リスクスコア計算用）
WEIGHTS = {
    "fire":             {"sns": 0.8, "weather": 0.1, "transport": 0.1},
    "flood":            {"sns": 0.5, "weather": 0.4, "transport": 0.1},
    "traffic_accident": {"sns": 0.6, "weather": 0.1, "transport": 0.3},
    "railway":          {"sns": 0.5, "weather": 0.1, "transport": 0.4},
    "unknown":          {"sns": 0.7, "weather": 0.15, "transport": 0.15},
}

# ─── アプリ初期化 ─────────────────────────────────────────────
app = FastAPI(title="Urban Safety AI Agent", version="0.1.0")

# CORS設定（ローカルネットワーク内のExpoアプリからアクセスできるように）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── データ読み込み ────────────────────────────────────────────
def load_events():
    with open(EVENTS_FILE, encoding="utf-8") as f:
        return json.load(f)

# ─── リスクスコア計算 ─────────────────────────────────────────
def calc_risk_score(event: dict) -> float:
    w = WEIGHTS.get(event["category"], WEIGHTS["unknown"])
    sns_score = event["source_count"] / 20  # 最大20件を1.0とする
    sns_score = min(sns_score, 1.0)
    score = (
        w["sns"] * sns_score
        + w["weather"] * event["weather_severity"]
        + w["transport"] * event["transport_severity"]
    )
    return round(min(score, 1.0), 3)

# ─── Ollamaへのリクエスト ─────────────────────────────────────
async def call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 300},
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        return resp.json()["response"].strip()

# ─── プロンプト生成 ───────────────────────────────────────────
def build_prompt(event: dict, risk_score: float) -> str:
    category_jp = {
        "fire": "火災",
        "flood": "浸水・洪水",
        "traffic_accident": "交通事故",
        "railway": "鉄道障害",
        "unknown": "不明",
    }.get(event["category"], "不明")

    return f"""あなたは都市安全AIエージェントです。以下のSNS投稿データを分析し、簡潔な日本語で警告レポートを生成してください。

【投稿内容】{event['text']}
【発生時刻】{event['timestamp']}
【発生場所】{event['location']}
【イベント種別】{category_jp}
【深刻度】{event['severity']}
【SNS投稿数】{event['source_count']}件
【気象リスク】{event['weather_severity']} (0〜1)
【交通リスク】{event['transport_severity']} (0〜1)
【総合リスクスコア】{risk_score} (0〜1)

以下の形式で回答してください：
1. 状況判断（1〜2文）
2. 推奨アクション（箇条書き2〜3点）
3. 注意事項（1文）

※ 模擬データに基づく参考情報です。実際の判断は公式機関の情報を確認してください。"""

# ─── エンドポイント ────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Urban Safety AI Agent is running"}

@app.get("/events")
def get_events():
    """イベント一覧を返す（リスクスコア付き）"""
    events = load_events()
    for ev in events:
        ev["risk_score"] = calc_risk_score(ev)
    # リスクスコア降順にソート
    events.sort(key=lambda x: x["risk_score"], reverse=True)
    return {"events": events}

@app.get("/events/{event_id}")
def get_event(event_id: str):
    """特定イベントの詳細を返す"""
    events = load_events()
    event = next((e for e in events if e["id"] == event_id), None)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    event["risk_score"] = calc_risk_score(event)
    return event

@app.post("/analyze/{event_id}")
async def analyze_event(event_id: str):
    """Ollamaを使ってイベントを分析し警告レポートを返す"""
    events = load_events()
    event = next((e for e in events if e["id"] == event_id), None)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    risk_score = calc_risk_score(event)
    prompt = build_prompt(event, risk_score)

    try:
        analysis = await call_ollama(prompt)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")

    return {
        "event_id": event_id,
        "risk_score": risk_score,
        "analysis": analysis,
        "model": OLLAMA_MODEL,
    }


@app.get("/health")
async def health_check():
    """Ollamaの接続確認"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
        return {"status": "ok", "ollama": "connected", "models": models}
    except Exception as e:
        return {"status": "error", "ollama": "disconnected", "detail": str(e)}

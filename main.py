"""
Urban Safety AI Agent - Backend v2
FastAPI + SQLite + Ollama (Llama)
"""

import sqlite3
import httpx
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2-vision:latest"
DB_FILE      = Path(__file__).parent / "urban_safety.db"

WEIGHTS = {
    "fire":             {"sns": 0.8, "weather": 0.1, "transport": 0.1},
    "flood":            {"sns": 0.5, "weather": 0.4, "transport": 0.1},
    "traffic_accident": {"sns": 0.6, "weather": 0.1, "transport": 0.3},
    "railway":          {"sns": 0.5, "weather": 0.1, "transport": 0.4},
    "noise":            {"sns": 0.0, "weather": 0.0, "transport": 0.0},
}

app = FastAPI(title="Urban Safety AI Agent", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def get_db():
    if not DB_FILE.exists():
        raise HTTPException(status_code=503, detail="DBが見つかりません。init_db.py を実行してください。")
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def row_to_dict(row) -> dict:
    d = dict(row)
    d["is_noise"] = bool(d.get("is_noise", 0))
    return d

def calc_risk_score(event: dict) -> float:
    if event.get("is_noise"):
        return 0.0
    w = WEIGHTS.get(event["category"], {"sns": 0.7, "weather": 0.15, "transport": 0.15})
    sns_score = min(event["source_count"] / 20, 1.0)
    score = w["sns"] * sns_score + w["weather"] * event["weather_severity"] + w["transport"] * event["transport_severity"]
    return round(min(score, 1.0), 3)

async def call_ollama(prompt: str) -> str:
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.3, "num_predict": 300}}
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        return resp.json()["response"].strip()

def build_prompt(event: dict, risk_score: float) -> str:
    category_jp = {"fire": "火災", "flood": "浸水・洪水", "traffic_accident": "交通事故", "railway": "鉄道障害"}.get(event["category"], "不明")
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

@app.get("/")
def root():
    return {"status": "ok", "version": "2.0.0"}

@app.get("/events")
def get_events(
    limit: int = Query(default=20, ge=1, le=100),
    category: Optional[str] = Query(default=None),
    severity: Optional[str] = Query(default=None),
    include_noise: bool = Query(default=False),
):
    conn = get_db()
    cur = conn.cursor()
    conditions, params = [], []
    if not include_noise:
        conditions.append("is_noise = 0")
    if category:
        conditions.append("category = ?")
        params.append(category)
    if severity:
        conditions.append("severity = ?")
        params.append(severity)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)
    cur.execute(f"SELECT * FROM events {where} ORDER BY risk_score DESC, timestamp DESC LIMIT ?", params)
    rows = [row_to_dict(r) for r in cur.fetchall()]
    conn.close()
    return {"events": rows, "count": len(rows)}

@app.get("/events/{event_id_param}")
def get_event(event_id_param: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM events WHERE id = ?", (event_id_param,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Event not found")
    return row_to_dict(row)

@app.post("/analyze/{event_id_param}")
async def analyze_event(event_id_param: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM events WHERE id = ?", (event_id_param,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Event not found")
    event = row_to_dict(row)
    risk_score = event.get("risk_score") or calc_risk_score(event)
    prompt = build_prompt(event, risk_score)
    try:
        analysis = await call_ollama(prompt)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")
    return {"event_id": event_id_param, "risk_score": risk_score, "analysis": analysis, "model": OLLAMA_MODEL}

@app.get("/stats")
def get_stats():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM events WHERE is_noise = 0")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM events WHERE is_noise = 1")
    noise = cur.fetchone()[0]
    cur.execute("SELECT category, severity, COUNT(*) as cnt FROM events WHERE is_noise = 0 GROUP BY category, severity ORDER BY category")
    breakdown = [dict(r) for r in cur.fetchall()]
    conn.close()
    return {"total_events": total, "noise_posts": noise, "breakdown": breakdown}

@app.get("/health")
async def health_check():
    db_ok = DB_FILE.exists()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
        ollama_status = "connected"
    except Exception as e:
        models = []
        ollama_status = f"error: {e}"
    return {"status": "ok" if db_ok and ollama_status == "connected" else "degraded", "db": "found" if db_ok else "missing", "ollama": ollama_status, "models": models}

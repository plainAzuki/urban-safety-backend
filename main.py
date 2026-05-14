"""Urban Safety AI Agent - Backend v2."""

import os
import sqlite3
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from official_sources import official_summary, signals_from_event

AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama").lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2-vision:latest")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DB_FILE = Path(__file__).parent / "urban_safety.db"

WEIGHTS = {
    "fire":             {"sns": 0.8, "weather": 0.1, "transport": 0.1},
    "flood":            {"sns": 0.5, "weather": 0.4, "transport": 0.1},
    "traffic_accident": {"sns": 0.6, "weather": 0.1, "transport": 0.3},
    "railway":          {"sns": 0.5, "weather": 0.1, "transport": 0.4},
    "noise":            {"sns": 0.0, "weather": 0.0, "transport": 0.0},
}

app = FastAPI(title="Urban Safety AI Agent", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CREATE_OFFICIAL_TABLE = """
CREATE TABLE IF NOT EXISTS official_observations (
    id          TEXT PRIMARY KEY,
    event_id    TEXT NOT NULL,
    source      TEXT NOT NULL,
    label       TEXT NOT NULL,
    severity    REAL NOT NULL,
    status      TEXT,
    detail      TEXT,
    observed_at TEXT,
    created_at  TEXT DEFAULT (datetime('now', 'localtime'))
);
CREATE INDEX IF NOT EXISTS idx_official_event_id ON official_observations(event_id);
"""


def get_db():
    if not DB_FILE.exists():
        raise HTTPException(status_code=503, detail="DBが見つかりません。init_db.py を実行してください。")
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    ensure_official_table(conn)
    return conn


def ensure_official_table(conn):
    conn.executescript(CREATE_OFFICIAL_TABLE)
    conn.commit()


def row_to_dict(row) -> dict:
    d = dict(row)
    d["is_noise"] = bool(d.get("is_noise", 0))
    return d


def parse_timestamp(value: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def risk_level(score: float) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def calc_risk_score(event: dict) -> float:
    if event.get("is_noise"):
        return 0.0
    w = WEIGHTS.get(event["category"], {"sns": 0.7, "weather": 0.15, "transport": 0.15})
    sns_score = min(event["source_count"] / 20, 1.0)
    score = w["sns"] * sns_score + w["weather"] * event["weather_severity"] + w["transport"] * event["transport_severity"]
    return round(min(score, 1.0), 3)


def load_official_signals(conn, event_id: str) -> list[dict]:
    cur = conn.cursor()
    cur.execute("""
        SELECT source, label, severity, status, detail, observed_at
        FROM official_observations
        WHERE event_id = ?
        ORDER BY severity DESC, observed_at DESC
    """, (event_id,))
    return [dict(row) for row in cur.fetchall()]


def save_official_signals(conn, event: dict, signals: list[dict]) -> int:
    inserted = 0
    event_key = event.get("event_id") or event["id"]
    cur = conn.cursor()
    for index, signal in enumerate(signals):
        observation_id = f"{event_key}:{signal['source']}:{signal['label']}:{index}"
        cur.execute("""
            INSERT OR REPLACE INTO official_observations
                (id, event_id, source, label, severity, status, detail, observed_at)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            observation_id,
            event_key,
            signal["source"],
            sign4al["label"],
            signal["severity"],
            signal.get("status"),
            signal.get("detail"),
            signal.get("observed_at"),
        ))
        inserted += 1
    conn.commit()
    return inserted


def get_official_signals(event: dict, conn=None) -> list[dict]:
    event_key = event.get("event_id") or event["id"]
    if conn is not None:
        stored = load_official_signals(conn, event_key)
        if stored:
            return stored
    elif DB_FILE.exists():
        local_conn = sqlite3.connect(DB_FILE)
        local_conn.row_factory = sqlite3.Row
        ensure_official_table(local_conn)
        stored = load_official_signals(local_conn, event_key)
        local_conn.close()
        if stored:
            return stored
    return signals_from_event(event)


def build_risk_item(rows: list[dict]) -> dict:
    representative = max(rows, key=lambda r: (r["risk_score"], r["timestamp"]))
    latest = max(r["timestamp"] for r in rows)
    earliest = min(r["timestamp"] for r in rows)
    score = max(r["risk_score"] for r in rows)
    sns_posts = len(rows)
    official_signals = get_official_signals(representative)
    return {
        **representative,
        "latest_timestamp": latest,
        "earliest_timestamp": earliest,
        "risk_score": score,
        "risk_level": risk_level(score),
        "sns_posts": sns_posts,
        "official_signals": official_signals,
        "official_summary": official_summary(official_signals),
    }


async def call_ollama(prompt: str) -> str:
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.3, "num_predict": 300}}
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        return resp.json()["response"].strip()


async def call_openai(prompt: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY が設定されていません。")
    payload = {
        "model": OPENAI_MODEL,
        "input": prompt,
        "temperature": 0.3,
        "max_output_tokens": 500,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post("https://api.openai.com/v1/responses", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if data.get("output_text"):
            return data["output_text"].strip()
        texts = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    texts.append(content.get("text", ""))
        return "\n".join(texts).strip()


async def call_ai(prompt: str) -> tuple[str, str]:
    if AI_PROVIDER == "openai":
        return await call_openai(prompt), OPENAI_MODEL
    return await call_ollama(prompt), OLLAMA_MODEL


def build_prompt(event: dict, risk_score: float) -> str:
    category_jp = {"fire": "火災", "flood": "浸水・洪水", "traffic_accident": "交通事故", "railway": "鉄道障害"}.get(event["category"], "不明")
    official = get_official_signals(event)
    official_text = " / ".join(f"{s['source']}:{s['label']}({s['severity']})" for s in official) or "目立った公式リスク信号なし"
    return f"""あなたは都市安全AIエージェントです。SNS模擬投稿と公式API由来の信号を統合し、簡潔な日本語で警告レポートを生成してください。

【投稿内容】{event['text']}
【発生時刻】{event['timestamp']}
【発生場所】{event['location']}
【イベント種別】{category_jp}
【深刻度】{event['severity']}
【SNS投稿数】{event['source_count']}件
【公式信号】{official_text}
【気象リスク】{event['weather_severity']} (0〜1)
【交通・鉄道リスク】{event['transport_severity']} (0〜1)
【総合リスクスコア】{risk_score} (0〜1)

以下の形式で回答してください：
1. 状況判断（1〜2文）
2. 推奨アクション（箇条書き2〜3点）
3. 注意事項（1文）

※ 模擬データに基づく参考情報です。実際の判断は公式機関の情報を確認してください。"""

@app.get("/")
def root():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/risks")
def get_risks(
    hours: int = Query(default=24, ge=1, le=168),
    limit: int = Query(default=20, ge=1, le=100),
    category: Optional[str] = Query(default=None),
):
    conn = get_db()
    cur = conn.cursor()
    conditions, params = ["is_noise = 0"], []
    if category:
        conditions.append("category = ?")
        params.append(category)
    cur.execute(f"SELECT * FROM events WHERE {' AND '.join(conditions)}", params)
    rows = [row_to_dict(r) for r in cur.fetchall()]
    conn.close()

    parsed = [r for r in rows if parse_timestamp(r["timestamp"])]
    if not parsed:
        return {"risks": [], "count": 0, "hours": hours}

    latest_at = max(parse_timestamp(r["timestamp"]) for r in parsed)
    since = latest_at - timedelta(hours=hours)
    groups = {}
    for row in parsed:
        ts = parse_timestamp(row["timestamp"])
        if ts and ts >= since:
            groups.setdefault(row["event_id"] or row["id"], []).append(row)

    risks = [build_risk_item(items) for items in groups.values()]
    risks.sort(key=lambda r: (r["risk_score"], r["latest_timestamp"]), reverse=True)
    return {"risks": risks[:limit], "count": len(risks[:limit]), "hours": hours, "latest_timestamp": latest_at.strftime("%Y-%m-%d %H:%M")}


@app.get("/dashboard")
def get_dashboard(
    hours: int = Query(default=24, ge=1, le=168),
    category: Optional[str] = Query(default=None),
):
    risks = get_risks(hours=hours, limit=100, category=category)["risks"]
    category_counts = Counter(r["category"] for r in risks)
    level_counts = Counter(r["risk_level"] for r in risks)
    top_risk = risks[0] if risks else None
    return {
        "hours": hours,
        "risk_count": len(risks),
        "top_risk": top_risk,
        "category_counts": dict(category_counts),
        "level_counts": {
            "high": level_counts.get("high", 0),
            "medium": level_counts.get("medium", 0),
            "low": level_counts.get("low", 0),
        },
        "risks": risks,
    }


@app.post("/official/sync")
def sync_official_observations(
    hours: int = Query(default=24, ge=1, le=168),
    category: Optional[str] = Query(default=None),
):
    risks = get_risks(hours=hours, limit=100, category=category)["risks"]
    conn = get_db()
    saved = 0
    for risk in risks:
        # 現段階では模擬値を公式API相当の形式に変換して保存する。
        saved += save_official_signals(conn, risk, signals_from_event(risk))
    conn.close()
    return {"synced_events": len(risks), "saved_signals": saved, "hours": hours, "category": category}


@app.get("/official/context/{event_id_param}")
def get_official_context(event_id_param: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM events WHERE id = ? OR event_id = ? ORDER BY risk_score DESC LIMIT 1", (event_id_param, event_id_param))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Event not found")
    event = row_to_dict(row)
    signals = get_official_signals(event, conn)
    if not signals:
        signals = signals_from_event(event)
    conn.close()
    return {
        "event_id": event.get("event_id") or event["id"],
        "signals": signals,
        "summary": official_summary(signals),
    }


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
        analysis, model = await call_ai(prompt)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"AI error: {str(e)}")
    return {
        "event_id": event_id_param,
        "risk_score": risk_score,
        "official_signals": get_official_signals(event),
        "analysis": analysis,
        "model": model,
        "provider": AI_PROVIDER,
    }

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
    models = []
    ai_status = "not_checked"
    if AI_PROVIDER == "ollama":
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(OLLAMA_URL.replace("/api/generate", "/api/tags"))
                models = [m["name"] for m in resp.json().get("models", [])]
            ai_status = "connected"
        except Exception as e:
            ai_status = f"error: {e}"
    elif AI_PROVIDER == "openai":
        ai_status = "configured" if OPENAI_API_KEY else "missing_api_key"
        models = [OPENAI_MODEL]
    return {"status": "ok" if db_ok and ai_status in {"connected", "configured"} else "degraded", "db": "found" if db_ok else "missing", "ai_provider": AI_PROVIDER, "ai": ai_status, "models": models}

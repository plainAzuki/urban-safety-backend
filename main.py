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

from official_sources import fetch_jma_aichi_signals, official_summary, signals_from_event, source_catalog

AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama").lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2-vision:latest")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DB_FILE = Path(__file__).parent / "urban_safety.db"

# カテゴリごとに「SNS量・気象・交通」の寄与を変え、現実の判断に近い重み付けにする。
WEIGHTS = {
    "fire":             {"sns": 0.8, "weather": 0.1, "transport": 0.1},
    "flood":            {"sns": 0.5, "weather": 0.4, "transport": 0.1},
    "traffic_accident": {"sns": 0.6, "weather": 0.1, "transport": 0.3},
    "railway":          {"sns": 0.5, "weather": 0.1, "transport": 0.4},
    "noise":            {"sns": 0.0, "weather": 0.0, "transport": 0.0},
}

CATEGORY_LABELS = {
    "fire": "火災",
    "flood": "浸水・洪水",
    "traffic_accident": "交通事故",
    "railway": "鉄道障害",
}

ACTION_GUIDES = {
    "fire": ["煙の方向を避けて移動する", "消防・自治体の発表を確認する", "現場周辺への接近を控える"],
    "flood": ["低い道路や河川沿いを避ける", "避難情報と気象庁の警報を確認する", "徒歩・車での冠水地点通過を避ける"],
    "traffic_accident": ["現場付近の道路を迂回する", "救急・警察活動の妨げになる接近を避ける", "公共交通や別ルートを確認する"],
    "railway": ["鉄道会社の運行情報を確認する", "振替輸送やバス路線を検討する", "駅構内の混雑に注意する"],
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

CREATE_AREA_OFFICIAL_TABLE = """
CREATE TABLE IF NOT EXISTS official_area_observations (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    area        TEXT NOT NULL,
    label       TEXT NOT NULL,
    severity    REAL NOT NULL,
    status      TEXT,
    detail      TEXT,
    observed_at TEXT,
    created_at  TEXT DEFAULT (datetime('now', 'localtime'))
);
CREATE INDEX IF NOT EXISTS idx_area_observed_at ON official_area_observations(observed_at);
"""


def get_db():
    if not DB_FILE.exists():
        raise HTTPException(status_code=503, detail="DBが見つかりません。init_db.py を実行してください。")
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    ensure_official_table(conn)
    return conn


def ensure_official_table(conn):
    conn.executescript(CREATE_OFFICIAL_TABLE + CREATE_AREA_OFFICIAL_TABLE)
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


def risk_factors(event: dict) -> dict:
    """アプリ側で判断根拠を表示できるよう、リスクスコアの内訳を返す。"""
    if event.get("is_noise"):
        return {"sns": 0.0, "weather": 0.0, "transport": 0.0}
    weights = WEIGHTS.get(event["category"], {"sns": 0.7, "weather": 0.15, "transport": 0.15})
    sns_score = min(event["source_count"] / 20, 1.0)
    return {
        "sns": round(weights["sns"] * sns_score, 3),
        "weather": round(weights["weather"] * event["weather_severity"], 3),
        "transport": round(weights["transport"] * event["transport_severity"], 3),
    }


def explain_risk(event: dict, official_signals: list[dict]) -> str:
    """詳細画面の前に読める、短い日本語のリスク説明を生成する。"""
    factors = risk_factors(event)
    strongest = max(factors, key=factors.get)
    labels = {"sns": "SNS投稿の集中", "weather": "気象情報", "transport": "交通・鉄道情報"}
    official_level = official_summary(official_signals)["status"]
    official_text = "公式情報にも注意信号があります" if official_level != "normal" else "公式情報は通常監視です"
    return f"{labels[strongest]}の影響が最も大きく、{official_text}。"


def confidence_score(event: dict, official_signals: list[dict]) -> float:
    """SNS投稿数と公式信号の有無から、情報の確からしさを0〜1で表す。"""
    sns_confidence = min((event.get("source_count") or 1) / 15, 1.0)
    official_confidence = 0.0
    if official_signals:
        official_confidence = min(max(signal["severity"] for signal in official_signals), 1.0)
    score = 0.65 * sns_confidence + 0.35 * official_confidence
    return round(min(score, 1.0), 3)


def confidence_label(score: float) -> str:
    if score >= 0.75:
        return "高"
    if score >= 0.45:
        return "中"
    return "低"


def build_fallback_analysis(event: dict, risk_score: float, official_signals: list[dict]) -> str:
    """AI接続が不安定な場合でも、デモと利用者確認に必要な最低限の助言を返す。"""
    category = CATEGORY_LABELS.get(event["category"], "都市安全リスク")
    actions = ACTION_GUIDES.get(event["category"], ["周辺状況を確認する", "公式機関の情報を確認する", "無理な移動を避ける"])
    official_text = "公式信号あり" if official_signals else "強い公式信号なし"
    action_text = "\n".join(f"- {action}" for action in actions)
    return f"""1. 状況判断
{category}に関する投稿が集中しており、総合リスクは{risk_score:.2f}です。現在の公式情報は「{official_text}」として扱います。

2. 推奨アクション
{action_text}

3. 注意事項
これは模擬データに基づく参考情報のため、実際の判断では自治体・気象庁・交通機関の公式情報を確認してください。"""


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
            signal["label"],
            signal["severity"],
            signal.get("status"),
            signal.get("detail"),
            signal.get("observed_at"),
        ))
        inserted += 1
    conn.commit()
    return inserted


def save_area_official_signals(conn, signals: list[dict]) -> int:
    """イベント未紐づけの公式ライブ情報を、地域単位の観測値として保存する。"""
    inserted = 0
    cur = conn.cursor()
    for index, signal in enumerate(signals):
        detail_key = (signal.get("detail") or "").split(" / ")[-1]
        observation_id = f"{signal['source']}:{signal['area']}:{signal.get('observed_at')}:{detail_key or index}"
        cur.execute("""
            INSERT OR REPLACE INTO official_area_observations
                (id, source, area, label, severity, status, detail, observed_at)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            observation_id,
            signal["source"],
            signal["area"],
            signal["label"],
            signal["severity"],
            signal.get("status"),
            signal.get("detail"),
            signal.get("observed_at"),
        ))
        inserted += 1
    conn.commit()
    return inserted


def load_area_official_signals(conn, limit: int = 20) -> list[dict]:
    cur = conn.cursor()
    cur.execute("""
        SELECT source, area, label, severity, status, detail, observed_at, created_at
        FROM official_area_observations
        ORDER BY observed_at DESC, created_at DESC
        LIMIT ?
    """, (limit,))
    return [dict(row) for row in cur.fetchall()]


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
    factors = risk_factors(representative)
    confidence = confidence_score(representative, official_signals)
    return {
        **representative,
        "latest_timestamp": latest,
        "earliest_timestamp": earliest,
        "risk_score": score,
        "risk_level": risk_level(score),
        "sns_posts": sns_posts,
        "risk_factors": factors,
        "risk_reason": explain_risk(representative, official_signals),
        "confidence_score": confidence,
        "confidence_label": confidence_label(confidence),
        "official_signals": official_signals,
        "official_summary": official_summary(official_signals),
    }


def count_live_official_observations() -> int:
    if not DB_FILE.exists():
        return 0
    conn = sqlite3.connect(DB_FILE)
    ensure_official_table(conn)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM official_area_observations")
    count = cur.fetchone()[0]
    conn.close()
    return count


def latest_live_official_observation() -> Optional[dict]:
    if not DB_FILE.exists():
        return None
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    ensure_official_table(conn)
    latest = load_area_official_signals(conn, limit=1)
    conn.close()
    return latest[0] if latest else None


def build_data_summary(risks: list[dict]) -> dict:
    """トップ画面でデータの鮮度と根拠を説明するための集計を作る。"""
    official_signal_count = sum(len(r.get("official_signals", [])) for r in risks)
    latest = max((r.get("latest_timestamp") for r in risks), default=None)
    earliest = min((r.get("earliest_timestamp") for r in risks), default=None)
    return {
        "event_count": len(risks),
        "official_signal_count": official_signal_count,
        "live_official_count": count_live_official_observations(),
        "latest_live_official": latest_live_official_observation(),
        "earliest_timestamp": earliest,
        "latest_timestamp": latest,
    }


def build_risk_timeline(risks: list[dict], hours: int) -> list[dict]:
    """過去期間の推移を、アプリで小さな棒グラフとして表示できる粒度にまとめる。"""
    timestamps = [parse_timestamp(r.get("latest_timestamp") or r["timestamp"]) for r in risks]
    timestamps = [ts for ts in timestamps if ts]
    if not timestamps:
        return []

    latest = max(timestamps)
    bucket_count = 6
    bucket_hours = max(hours / bucket_count, 1)
    buckets = [
        {
            "label": f"-{int(hours - index * bucket_hours)}h",
            "count": 0,
            "high_count": 0,
            "max_score": 0.0,
        }
        for index in range(bucket_count)
    ]

    for risk in risks:
        ts = parse_timestamp(risk.get("latest_timestamp") or risk["timestamp"])
        if not ts:
            continue
        elapsed_hours = max((latest - ts).total_seconds() / 3600, 0)
        index = min(int(elapsed_hours / bucket_hours), bucket_count - 1)
        bucket = buckets[bucket_count - 1 - index]
        bucket["count"] += 1
        bucket["high_count"] += 1 if risk["risk_level"] == "high" else 0
        bucket["max_score"] = max(bucket["max_score"], risk["risk_score"])

    for bucket in buckets:
        bucket["max_score"] = round(bucket["max_score"], 3)
    return buckets


def current_ai_config() -> dict:
    """現在選択されているAI接続先を、ヘルスチェックや説明画面で使いやすい形にする。"""
    return {
        "provider": AI_PROVIDER,
        "model": OPENAI_MODEL if AI_PROVIDER == "openai" else OLLAMA_MODEL,
        "openai_ready": bool(OPENAI_API_KEY) if AI_PROVIDER == "openai" else None,
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
    confidence = confidence_score(event, official)
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
【情報信頼度】{confidence} (0〜1)

以下の形式で回答してください：
1. 状況判断（1〜2文）
2. 推奨アクション（箇条書き2〜3点）
3. 注意事項（1文）

※ 模擬データに基づく参考情報です。実際の判断は公式機関の情報を確認してください。"""

@app.get("/")
def root():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/system/overview")
def get_system_overview():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM events WHERE is_noise = 0")
    event_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM events WHERE is_noise = 1")
    noise_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM official_observations")
    official_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM official_area_observations")
    live_official_count = cur.fetchone()[0]
    conn.close()
    return {
        "concept": "SNS模擬投稿と公式情報を統合し、都市安全リスクを説明可能な形で提示する卒業制作プロトタイプ",
        "pipeline": [
            "SNS風の模擬投稿を生成・保存",
            "気象・鉄道・交通に相当する公式信号をイベント単位で統合",
            "カテゴリ別の重みで総合リスクを算出",
            "アプリで一覧・時間推移・詳細助言を表示",
            "LLMで利用者向けの行動提案を生成",
        ],
        "ai": current_ai_config(),
        "database": {
            "event_count": event_count,
            "noise_count": noise_count,
            "official_observation_count": official_count,
            "live_official_observation_count": live_official_count,
        },
        "weights": WEIGHTS,
    }


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
    data_summary = build_data_summary(risks)
    return {
        "hours": hours,
        "basis": "SNS模擬投稿、気象リスク、交通・鉄道リスクを統合した参考評価",
        "data_summary": data_summary,
        "risk_timeline": build_risk_timeline(risks, hours),
        "ai_config": current_ai_config(),
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


@app.get("/official/sources")
def get_official_sources():
    return {"sources": source_catalog()}


@app.get("/official/live")
def get_live_official_observations(limit: int = Query(default=20, ge=1, le=100)):
    conn = get_db()
    observations = load_area_official_signals(conn, limit)
    conn.close()
    return {"observations": observations, "count": len(observations)}


@app.get("/official/live/weather")
async def get_live_weather_signals(limit: int = Query(default=20, ge=1, le=100)):
    signals = await fetch_jma_aichi_signals(limit=limit)
    return {"source": "気象庁防災情報XML", "signals": signals, "count": len(signals)}


@app.post("/official/live/sync")
async def sync_live_official_observations(limit: int = Query(default=20, ge=1, le=100)):
    signals = await fetch_jma_aichi_signals(limit=limit)
    conn = get_db()
    conn.execute("DELETE FROM official_area_observations WHERE source = ?", ("気象庁防災情報XML",))
    saved = save_area_official_signals(conn, signals)
    conn.close()
    return {"fetched": len(signals), "saved": saved, "source": "気象庁防災情報XML"}


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
    official_signals = get_official_signals(event)
    confidence = confidence_score(event, official_signals)
    prompt = build_prompt(event, risk_score)
    try:
        analysis, model = await call_ai(prompt)
    except Exception as e:
        analysis = build_fallback_analysis(event, risk_score, official_signals)
        model = "rule-based-fallback"
        ai_error = str(e)
    else:
        ai_error = None
    return {
        "event_id": event_id_param,
        "risk_score": risk_score,
        "risk_factors": risk_factors(event),
        "risk_reason": explain_risk(event, official_signals),
        "confidence_score": confidence,
        "confidence_label": confidence_label(confidence),
        "official_signals": official_signals,
        "analysis": analysis,
        "model": model,
        "provider": AI_PROVIDER,
        "ai_error": ai_error,
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
    return {
        "status": "ok" if db_ok and ai_status in {"connected", "configured"} else "degraded",
        "db": "found" if db_ok else "missing",
        "ai_provider": AI_PROVIDER,
        "ai": ai_status,
        "ai_config": current_ai_config(),
        "models": models,
    }

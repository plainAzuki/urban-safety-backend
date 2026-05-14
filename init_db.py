"""
Step 2: SQLiteデータベース初期化・データ投入スクリプト
generate_data.py で生成した mock_events.json を
SQLiteデータベースに格納する。

使い方:
    python init_db.py
"""

import json
import sqlite3
from pathlib import Path

DB_FILE    = "urban_safety.db"
JSON_FILE  = "mock_events.json" #模擬データ

# ─── テーブル作成 ─────────────────────────────────────────────
CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    id                  TEXT PRIMARY KEY,
    timestamp           TEXT NOT NULL,
    text                TEXT NOT NULL,
    location            TEXT,
    lat                 REAL,
    lng                 REAL,
    category            TEXT,
    severity            TEXT,
    weather_severity    REAL DEFAULT 0.0,
    transport_severity  REAL DEFAULT 0.0,
    source_count        INTEGER DEFAULT 1,
    event_id            TEXT,
    is_noise            INTEGER DEFAULT 0,   -- SQLiteはBOOLEANなし、0/1で代用
    risk_score          REAL DEFAULT 0.0,
    created_at          TEXT DEFAULT (datetime('now', 'localtime'))
);
"""

CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_event_id  ON events(event_id);
CREATE INDEX IF NOT EXISTS idx_category  ON events(category);
CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_is_noise  ON events(is_noise);
"""

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

CREATE_AI_ANALYSIS_TABLE = """
CREATE TABLE IF NOT EXISTS ai_analyses (
    id          TEXT PRIMARY KEY,
    event_id    TEXT NOT NULL,
    provider    TEXT NOT NULL,
    model       TEXT NOT NULL,
    risk_score  REAL NOT NULL,
    analysis    TEXT NOT NULL,
    ai_error    TEXT,
    created_at  TEXT DEFAULT (datetime('now', 'localtime'))
);
CREATE INDEX IF NOT EXISTS idx_ai_analyses_event_id ON ai_analyses(event_id);
"""

# ─── リスクスコア計算（main.pyと同じロジック） ──────────────
WEIGHTS = {
    "fire":             {"sns": 0.8, "weather": 0.1, "transport": 0.1},
    "flood":            {"sns": 0.5, "weather": 0.4, "transport": 0.1},
    "traffic_accident": {"sns": 0.6, "weather": 0.1, "transport": 0.3},
    "railway":          {"sns": 0.5, "weather": 0.1, "transport": 0.4},
    "noise":            {"sns": 0.0, "weather": 0.0, "transport": 0.0},
}

def calc_risk_score(event: dict) -> float:
    if event.get("is_noise"):
        return 0.0
    w = WEIGHTS.get(event["category"], {"sns": 0.7, "weather": 0.15, "transport": 0.15})
    sns_score = min(event["source_count"] / 20, 1.0)
    score = (
        w["sns"] * sns_score
        + w["weather"] * event["weather_severity"]
        + w["transport"] * event["transport_severity"]
    )
    return round(min(score, 1.0), 3)

# ─── メイン処理 ──────────────────────────────────────────────
def main():
    # JSONファイル確認
    if not Path(JSON_FILE).exists():
        print(f"❌ {JSON_FILE} が見つかりません。先に generate_data.py を実行してください。")
        return

    with open(JSON_FILE, encoding="utf-8") as f:
        events = json.load(f)

    # DB接続・テーブル作成
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.executescript(CREATE_TABLE + CREATE_INDEX + CREATE_OFFICIAL_TABLE + CREATE_AREA_OFFICIAL_TABLE + CREATE_AI_ANALYSIS_TABLE)
    conn.commit()
    print(f"✅ テーブル作成完了: {DB_FILE}")

    # データ投入
    inserted = 0
    skipped  = 0
    for ev in events:
        ev["risk_score"] = calc_risk_score(ev)
        ev["is_noise"]   = 1 if ev.get("is_noise") else 0
        try:
            cur.execute("""
                INSERT INTO events
                    (id, timestamp, text, location, lat, lng,
                     category, severity, weather_severity, transport_severity,
                     source_count, event_id, is_noise, risk_score)
                VALUES
                    (:id, :timestamp, :text, :location, :lat, :lng,
                     :category, :severity, :weather_severity, :transport_severity,
                     :source_count, :event_id, :is_noise, :risk_score)
            """, ev)
            inserted += 1
        except sqlite3.IntegrityError:
            skipped += 1  # 重複IDはスキップ

    conn.commit()
    conn.close()

    print(f"✅ データ投入完了: {inserted}件挿入 / {skipped}件スキップ")

    # 確認クエリ
    conn = sqlite3.connect(DB_FILE)
    cur  = conn.cursor()
    cur.execute("SELECT category, COUNT(*) as cnt FROM events GROUP BY category ORDER BY cnt DESC")
    rows = cur.fetchall()
    conn.close()

    print("\n─── カテゴリ別集計 ───")
    for row in rows:
        print(f"  {row[0]:20s}: {row[1]}件")

if __name__ == "__main__":
    main()

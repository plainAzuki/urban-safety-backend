"""SQLite のテーブル定義とCRUD。

卒論上は official_area_observations が Evidence DB、
answer_verifications が AI回答検証ログに対応する。
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4

from config import DB_FILE, OFFICIAL_FETCH_MIN_INTERVAL_MINUTES, OFFICIAL_HISTORY_PER_SOURCE
from official_sources import official_source_names


CREATE_AREA_OFFICIAL_TABLE = """
CREATE TABLE IF NOT EXISTS official_area_observations (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    source_url  TEXT,
    area        TEXT NOT NULL,
    label       TEXT NOT NULL,
    severity    REAL NOT NULL,
    status      TEXT,
    detail      TEXT,
    observed_at TEXT,
    created_at  TEXT DEFAULT (datetime('now', 'localtime'))
);
CREATE INDEX IF NOT EXISTS idx_area_observed_at ON official_area_observations(observed_at);
CREATE INDEX IF NOT EXISTS idx_area_source_created ON official_area_observations(source, created_at);
"""

CREATE_ANSWER_VERIFICATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS answer_verifications (
    id             TEXT PRIMARY KEY,
    question       TEXT NOT NULL,
    draft_answer   TEXT NOT NULL,
    visible_answer TEXT,
    verdict        TEXT NOT NULL,
    display_policy TEXT NOT NULL,
    warning        TEXT,
    reasons_json   TEXT,
    model          TEXT NOT NULL,
    provider       TEXT NOT NULL,
    ai_error       TEXT,
    created_at     TEXT DEFAULT (datetime('now', 'localtime'))
);
CREATE INDEX IF NOT EXISTS idx_answer_verifications_created ON answer_verifications(created_at);
"""

DROP_OBSOLETE_TABLES = """
DROP TABLE IF EXISTS events;
DROP TABLE IF EXISTS official_observations;
DROP TABLE IF EXISTS ai_analyses;
DROP TABLE IF EXISTS experiment_results;
DROP TABLE IF EXISTS experiment_runs;
"""


def get_db():
    """DB接続を返し、起動時のテーブル不足も補正する。"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    ensure_tables(conn)
    return conn


def ensure_tables(conn):
    """現行スキーマを作成する。既存DBでも不足列を追加する。"""
    conn.executescript(
        DROP_OBSOLETE_TABLES
        + CREATE_AREA_OFFICIAL_TABLE
        + CREATE_ANSWER_VERIFICATIONS_TABLE
    )
    ensure_column(conn, "official_area_observations", "source_url", "TEXT")
    conn.commit()


def ensure_column(conn, table: str, column: str, definition: str) -> None:
    """古いDBファイルに不足している列だけ追加する。"""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def official_signal_row_to_dict(row) -> dict:
    """API返却用にRowをdictへ変換する。"""
    return dict(row)


def parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    """DBや公式フィードの時刻文字列を比較可能な datetime に変換する。"""
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def load_area_official_signals(conn, limit: int = 20) -> list[dict]:
    """Evidence DB から新しい順に公式情報を取得する。"""
    sources = official_source_names()
    placeholders = ",".join("?" for _ in sources)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT source, source_url, area, label, severity, status, detail, observed_at, created_at
        FROM official_area_observations
        WHERE source IN ({placeholders})
        ORDER BY observed_at DESC, created_at DESC
        LIMIT ?
    """, (*sources, limit))
    return [official_signal_row_to_dict(row) for row in cur.fetchall()]


def load_latest_official_signals_by_source(conn, limit_per_source: int = 1) -> list[dict]:
    """情報源ごとの最新公式情報を取得する。ダッシュボード用。"""
    sources = official_source_names()
    placeholders = ",".join("?" for _ in sources)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT source, source_url, area, label, severity, status, detail, observed_at, created_at
        FROM (
            SELECT
                source, source_url, area, label, severity, status, detail, observed_at, created_at,
                ROW_NUMBER() OVER (
                    PARTITION BY source
                    ORDER BY datetime(created_at) DESC, observed_at DESC, id DESC
                ) AS row_number
            FROM official_area_observations
            WHERE source IN ({placeholders})
        )
        WHERE row_number <= ?
        ORDER BY source ASC, datetime(created_at) DESC, observed_at DESC
    """, (*sources, limit_per_source))
    return [official_signal_row_to_dict(row) for row in cur.fetchall()]


def load_official_history_for_source(conn, source: str, limit: int = OFFICIAL_HISTORY_PER_SOURCE) -> list[dict]:
    """指定した情報源の履歴を取得する。詳細モーダル用。"""
    cur = conn.cursor()
    cur.execute("""
        SELECT source, source_url, area, label, severity, status, detail, observed_at, created_at
        FROM official_area_observations
        WHERE source = ?
        ORDER BY datetime(created_at) DESC, observed_at DESC, id DESC
        LIMIT ?
    """, (source, limit))
    return [official_signal_row_to_dict(row) for row in cur.fetchall()]


def trim_official_history(conn, per_source: int = OFFICIAL_HISTORY_PER_SOURCE) -> int:
    """情報源ごとに必要な件数だけ残し、DBを肥大化させない。"""
    sources = official_source_names()
    deleted = 0
    for source in sources:
        cur = conn.execute("""
            DELETE FROM official_area_observations
            WHERE source = ?
              AND id NOT IN (
                SELECT id
                FROM official_area_observations
                WHERE source = ?
                ORDER BY datetime(created_at) DESC, observed_at DESC, id DESC
                LIMIT ?
              )
        """, (source, source, per_source))
        deleted += cur.rowcount
    conn.commit()
    return deleted


def save_area_official_signals(conn, signals: list[dict]) -> int:
    """正規化済み公式情報を Evidence DB に保存する。"""
    inserted = 0
    cur = conn.cursor()
    for index, signal in enumerate(signals):
        detail_key = (signal.get("detail") or "").split(" / ")[-1]
        observation_id = f"{signal['source']}:{signal['area']}:{signal.get('observed_at')}:{detail_key or index}"
        cur.execute("""
            INSERT OR REPLACE INTO official_area_observations
                (id, source, source_url, area, label, severity, status, detail, observed_at)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            observation_id,
            signal["source"],
            signal.get("source_url"),
            signal["area"],
            signal["label"],
            signal["severity"],
            signal.get("status"),
            signal.get("detail"),
            signal.get("observed_at"),
        ))
        inserted += 1
    conn.commit()
    trim_official_history(conn)
    return inserted


def official_refresh_due(conn) -> tuple[bool, Optional[str]]:
    """公式情報を再取得すべきか、最後の保存時刻から判断する。"""
    sources = official_source_names()
    placeholders = ",".join("?" for _ in sources)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT MAX(created_at)
        FROM official_area_observations
        WHERE source IN ({placeholders})
    """, sources)
    latest_text = cur.fetchone()[0]
    latest = parse_timestamp(latest_text)
    if latest is None:
        return True, latest_text
    age = datetime.now() - latest
    return age >= timedelta(minutes=OFFICIAL_FETCH_MIN_INTERVAL_MINUTES), latest_text


def clear_area_official_sources(conn) -> None:
    """現状では履歴保持のみ行う。将来の全削除方針変更に備えた関数。"""
    trim_official_history(conn)


def delete_answer_cache() -> int:
    """回答検証ログを削除する。公式情報更新時の古い回答無効化に使う。"""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM answer_verifications")
    deleted = cur.rowcount
    conn.commit()
    conn.close()
    return deleted


def save_answer_verification(
    question: str,
    draft_answer: str,
    visible_answer: Optional[str],
    verification: dict,
    model: str,
    provider: str,
    ai_error: Optional[str],
) -> str:
    """AI回答とVerifier判定を保存する。"""
    answer_id = str(uuid4())
    conn = get_db()
    conn.execute("""
        INSERT INTO answer_verifications
            (id, question, draft_answer, visible_answer, verdict, display_policy, warning, reasons_json, model, provider, ai_error)
        VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        answer_id,
        question,
        draft_answer,
        visible_answer,
        verification["verdict"],
        verification["display_policy"],
        verification.get("warning"),
        json.dumps({
            "reasons": verification.get("reasons", []),
            "checked_claims": verification.get("checked_claims", []),
        }, ensure_ascii=False),
        model,
        provider,
        ai_error,
    ))
    conn.commit()
    conn.close()
    return answer_id

"""公式情報の取得、正規化、要約を担当するサービス層。"""

import re
from datetime import datetime
from typing import Optional

from ai_client import call_ai
from config import (
    AI_TIMEOUT_SECONDS,
    AI_NORMALIZER_MODEL,
    OFFICIAL_BACKGROUND_INTERVAL_MINUTES,
    OFFICIAL_FETCH_MIN_INTERVAL_MINUTES,
    OFFICIAL_HISTORY_PER_SOURCE,
    OFFICIAL_LLM_BATCH_SIZE,
    OFFICIAL_LLM_RAW_CHARS,
)
from db import (
    clear_area_official_sources,
    delete_answer_cache,
    get_db,
    load_area_official_signals,
    official_refresh_due,
    save_area_official_signals,
    trim_official_history,
)
from json_utils import extract_json_object
from official_sources import fetch_raw_official_records, now_text
from prompts import build_official_normalization_prompt


OFFICIAL_SIGNAL_STATUSES = {"通常", "情報", "注意", "警戒", "運休", "支障", "取得不可"}
STATUS_PRIORITY = {
    "取得不可": 0,
    "通常": 1,
    "情報": 2,
    "注意": 3,
    "支障": 4,
    "警戒": 5,
    "運休": 5,
}

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
TIMESTAMP_PATTERNS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d-%H-%M-%S",
    "%Y-%m-%d-%H-%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y%m%d%H%M%S",
    "%Y%m%d%H%M",
)
DATE_ONLY_PATTERN = re.compile(r"^\d{4}[-/年]?\d{1,2}[-/月]?\d{1,2}日?$")


def normalize_status(value: object, fallback: str = "情報") -> str:
    """日文statusへ正規化する。"""
    status = str(value or fallback).strip()
    return status if status in OFFICIAL_SIGNAL_STATUSES else fallback


def summarize_status(observations: list[dict]) -> str:
    """数値スコアを使わず、statusの優先度だけで全体状態を決める。"""
    statuses = [normalize_status(item.get("status"), fallback="情報") for item in observations]
    return max(statuses, key=lambda status: STATUS_PRIORITY.get(status, 0), default="通常")


def normalize_observed_at(value: object) -> str:
    """公式時刻があれば統一形式へ変換し、不明なら現在時刻を返す。"""
    if value:
        raw = str(value).strip()
        if raw:
            normalized = raw.replace("T", " ").replace("Z", "+00:00")
            normalized = re.sub(r"\s+", " ", normalized)
            if DATE_ONLY_PATTERN.fullmatch(normalized):
                return now_text()
            try:
                parsed = datetime.fromisoformat(normalized)
                if parsed.tzinfo is not None:
                    parsed = parsed.astimezone().replace(tzinfo=None)
                return parsed.replace(tzinfo=None).strftime(TIMESTAMP_FORMAT)
            except ValueError:
                pass
            for pattern in TIMESTAMP_PATTERNS:
                try:
                    return datetime.strptime(normalized, pattern).strftime(TIMESTAMP_FORMAT)
                except ValueError:
                    continue
            japanese_match = re.search(
                r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日\s*(\d{1,2})時(?:\s*(\d{1,2})分)?(?:\s*(\d{1,2})秒)?",
                normalized,
            )
            if japanese_match:
                year, month, day, hour, minute, second = japanese_match.groups()
                return datetime(
                    int(year),
                    int(month),
                    int(day),
                    int(hour),
                    int(minute or 0),
                    int(second or 0),
                ).strftime(TIMESTAMP_FORMAT)
    return now_text()


def live_official_summary(observations: Optional[list[dict]] = None) -> dict:
    """最新の公式情報群から、画面上部に出す全体状態を作る。"""
    if observations is None:
        conn = get_db()
        observations = load_area_official_signals(conn, limit=100)
        conn.close()
    if not observations:
        return {"count": 0, "status": "通常", "sources": []}
    return {
        "count": len(observations),
        "status": summarize_status(observations),
        "sources": sorted({item["source"] for item in observations}),
        "history_per_source": OFFICIAL_HISTORY_PER_SOURCE,
    }


def build_data_summary(observations: list[dict]) -> dict:
    """ダッシュボード用の件数・最新時刻をまとめる。"""
    latest = observations[0] if observations else None
    return {
        "official_count": len(observations),
        "live_official_count": len(observations),
        "live_official_summary": live_official_summary(observations),
        "latest_live_official": latest,
        "latest_timestamp": latest.get("observed_at") if latest else None,
    }


def clean_official_signal(signal: dict) -> dict:
    """LLM出力をDB保存可能な安全な形式へ丸める。"""
    status = normalize_status(signal.get("status"), fallback="情報")
    return {
        "source": str(signal.get("source") or "公式情報").strip()[:80],
        "source_url": str(signal.get("source_url") or signal.get("url") or "").strip()[:500],
        "area": str(signal.get("area") or "愛知県").strip()[:80],
        "label": str(signal.get("label") or "公式情報確認").strip()[:120],
        "severity": 0.0,
        "status": status,
        "detail": str(signal.get("detail") or "").strip()[:1200],
        "observed_at": normalize_observed_at(signal.get("observed_at")),
    }


def official_record_fallback(record: dict, exc: Exception) -> dict:
    """LLM正規化に失敗しても、取得した公式URL自体はEvidenceとして残す。"""
    return clean_official_signal({
        "source": record.get("source"),
        "source_url": record.get("source_url") or record.get("url"),
        "area": record.get("area"),
        "label": record.get("title") or "公式情報確認",
        "status": "取得不可" if "取得失敗" in (record.get("title") or "") else "情報",
        "detail": f"LLMによる公式情報の構造化に失敗しました。公式URL: {record.get('url')} / error: {type(exc).__name__}: {exc}",
        "observed_at": record.get("observed_at"),
    })


def chunk_records(records: list[dict], size: int) -> list[list[dict]]:
    """公式ページ原文をLLMへ渡す単位に分割する。"""
    chunk_size = max(1, size)
    return [records[index:index + chunk_size] for index in range(0, len(records), chunk_size)]


async def normalize_official_records_with_llm(records: list[dict]) -> tuple[list[dict], str, Optional[str]]:
    """公式原文を Evidence DB の共通レコードへ正規化する。"""
    if not records:
        return [], "no-official-records", None

    signals = []
    models = []
    errors = []
    batches = chunk_records(records, OFFICIAL_LLM_BATCH_SIZE)

    for batch_index, batch in enumerate(batches, start=1):
        prompt = build_official_normalization_prompt(batch)
        try:
            output, model = await call_ai(prompt, json_mode=True, model=AI_NORMALIZER_MODEL)
            data = extract_json_object(output)
            batch_signals = [
                clean_official_signal(signal)
                for signal in data.get("signals", [])
                if isinstance(signal, dict)
            ]
            for signal in batch_signals:
                if not signal.get("source_url"):
                    matched_record = next(
                        (record for record in batch if record.get("source") == signal.get("source")),
                        batch[0],
                    )
                    signal["source_url"] = matched_record.get("source_url") or matched_record.get("url") or ""
            if not batch_signals:
                raise ValueError("LLM出力にsignalsがありません。")
            signals.extend(batch_signals)
            models.append(model)
        except Exception as exc:
            errors.append(f"batch {batch_index}/{len(batches)}: {type(exc).__name__}: {exc}")
            signals.extend(official_record_fallback(record, exc) for record in batch)

    model_name = "+".join(sorted(set(models))) if models else "llm-normalization-fallback"
    error_text = " | ".join(errors) if errors else None
    return signals, model_name, error_text


async def run_official_sync(force: bool = False, limit: int = 20, source: str = "manual") -> dict:
    """公式情報の取得からDB保存までをまとめて実行する。"""
    conn = get_db()
    due, last_synced_at = official_refresh_due(conn)
    fetched = False
    llm_model = None
    llm_error = None
    raw_record_count = 0
    if force or due:
        raw_records = await fetch_raw_official_records(limit=limit)
        raw_record_count = len(raw_records)
        signals, llm_model, llm_error = await normalize_official_records_with_llm(raw_records)
        clear_area_official_sources(conn)
        saved = save_area_official_signals(conn, signals)
        fetched = True
    else:
        signals = load_area_official_signals(conn, limit=100)
        saved = 0
    deleted_answers = delete_answer_cache() if fetched else 0
    trim_official_history(conn)
    conn.close()
    return {
        "source": source,
        "fetched": len(signals) if fetched else 0,
        "raw_records": raw_record_count,
        "saved_area_signals": saved,
        "deleted_answer_cache": deleted_answers,
        "llm_model": llm_model,
        "llm_error": llm_error,
        "used_cache": not fetched,
        "last_synced_at": last_synced_at,
        "min_interval_minutes": OFFICIAL_FETCH_MIN_INTERVAL_MINUTES,
        "background_interval_minutes": OFFICIAL_BACKGROUND_INTERVAL_MINUTES,
        "history_per_source": OFFICIAL_HISTORY_PER_SOURCE,
        "llm_batch_size": OFFICIAL_LLM_BATCH_SIZE,
        "llm_raw_chars": OFFICIAL_LLM_RAW_CHARS,
        "ai_timeout_seconds": AI_TIMEOUT_SECONDS,
    }

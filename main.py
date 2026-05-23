"""Urban Safety official information backend.

このファイルは FastAPI のルーティングだけを担当する。
実処理は db.py / official_service.py / answer_service.py に分離している。
"""

import asyncio
import contextlib
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from ai_client import active_ai_model, current_ai_config
from answer_service import ask_official_question
from config import (
    AI_BASE_URL,
    AI_GENERATOR_MODEL,
    AI_MODEL,
    AI_NORMALIZER_MODEL,
    AI_PROVIDER,
    AI_VERIFIER_MODEL,
    DB_FILE,
    OFFICIAL_BACKGROUND_INTERVAL_MINUTES,
    OFFICIAL_HISTORY_PER_SOURCE,
)
from db import (
    delete_answer_cache,
    get_db,
    load_latest_official_signals_by_source,
    load_official_history_for_source,
)
from official_service import build_data_summary, live_official_summary, run_official_sync
from official_sources import fetch_jma_aichi_signals, official_status_catalog, source_catalog
from schemas import AskRequest


app = FastAPI(title="Urban Safety Official Agent", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# バックグラウンド同期はアプリ起動中だけ生きるタスクとして管理する。
official_sync_lock = asyncio.Lock()
official_background_task: Optional[asyncio.Task] = None


async def locked_official_sync(force: bool = False, limit: int = 20, source: str = "manual") -> dict:
    """同時に複数の公式取得が走らないようにするルート用ラッパー。"""
    async with official_sync_lock:
        return await run_official_sync(force=force, limit=limit, source=source)


async def official_background_loop() -> None:
    """一定間隔で公式情報を確認する。取得間隔の判定は service 側で行う。"""
    while True:
        try:
            await locked_official_sync(force=False, limit=1, source="background")
        except Exception as exc:
            print(f"official background sync failed: {type(exc).__name__}: {exc}")
        await asyncio.sleep(OFFICIAL_BACKGROUND_INTERVAL_MINUTES * 60)


@app.on_event("startup")
async def start_official_background_sync() -> None:
    """FastAPI 起動時に公式情報の定期確認を開始する。"""
    global official_background_task
    if official_background_task is None or official_background_task.done():
        official_background_task = asyncio.create_task(official_background_loop())


@app.on_event("shutdown")
async def stop_official_background_sync() -> None:
    """FastAPI 終了時にバックグラウンドタスクを停止する。"""
    if official_background_task is not None:
        official_background_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await official_background_task


@app.get("/")
def root():
    """疎通確認用の最小エンドポイント。"""
    return {"status": "ok", "version": "3.0.0"}


@app.get("/system/overview")
def get_system_overview():
    """卒論説明にも使える、システム全体像とDB状態の概要。"""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM official_area_observations")
    official_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM answer_verifications")
    answer_count = cur.fetchone()[0]
    conn.close()
    return {
        "concept": "公式情報を取得・構造化し、回答生成AgentとVerifier Agentで利用者表示を制御する都市安全情報プロトタイプ",
        "pipeline": [
            "公式情報源",
            "official/sync で取得",
            "LLMまたはルールで構造化",
            "公式情報DBへ保存",
            "ユーザー質問",
            "回答生成Agentで draft_answer を生成",
            "Verifier Agent が PASS / FAIL / NEEDS_REVIEW を判定",
            "SHOW / SHOW_WITH_WARNING / DO_NOT_SHOW に従ってUI表示",
        ],
        "ai": current_ai_config(),
        "official_statuses": official_status_catalog(),
        "database": {
            "official_observation_count": official_count,
            "answer_verification_count": answer_count,
        },
    }


@app.get("/dashboard")
def get_dashboard():
    """フロント画面用に、最新公式情報・全体状態・AI設定を返す。"""
    conn = get_db()
    observations = load_latest_official_signals_by_source(conn, limit_per_source=1)
    conn.close()
    summary = live_official_summary(observations)
    return {
        "basis": "公式サイト・公式APIから取得した情報のみを表示しています",
        "data_summary": build_data_summary(observations),
        "official_summary": summary,
        "official_observations": observations,
        "official_count": len(observations),
        "ai_config": current_ai_config(),
        "display_policy": "official_only",
    }


@app.post("/official/sync")
async def sync_official_observations(
    force: bool = Query(default=False),
    limit: int = Query(default=20, ge=1, le=100),
):
    """公式情報源へアクセスし、Evidence DB を更新する。"""
    return await locked_official_sync(force=force, limit=limit, source="manual")


@app.get("/official/sources")
def get_official_sources():
    """現在実装している公式情報源とstatus一覧。"""
    return {"sources": source_catalog(), "official_statuses": official_status_catalog()}


@app.get("/official/live")
def get_live_official_observations(
    limit: int = Query(default=20, ge=1, le=100),
    source: Optional[str] = Query(default=None),
):
    """Evidence DB に保存済みの公式情報を返す。"""
    conn = get_db()
    if source:
        observations = load_official_history_for_source(conn, source, limit=limit)
    else:
        observations = load_latest_official_signals_by_source(conn, limit_per_source=1)[:limit]
    conn.close()
    return {
        "source": source,
        "observations": observations,
        "count": len(observations),
        "summary": live_official_summary(observations),
        "history_per_source": OFFICIAL_HISTORY_PER_SOURCE,
    }


@app.get("/official/live/weather")
async def get_live_weather_signals(limit: int = Query(default=20, ge=1, le=100)):
    """気象庁XMLから愛知県関連の気象情報を直接取得する確認用API。"""
    signals = await fetch_jma_aichi_signals(limit=limit)
    return {"source": "気象庁防災情報XML", "signals": signals, "count": len(signals)}


@app.post("/official/live/sync")
async def sync_live_official_observations(
    limit: int = Query(default=20, ge=1, le=100),
    force: bool = Query(default=False),
):
    """旧UI互換の公式情報同期API。内部では /official/sync と同じ処理。"""
    return await locked_official_sync(force=force, limit=limit, source="manual-live")


@app.post("/ask")
async def ask_official_agent(request: AskRequest):
    """ユーザー質問に対して、Generator と Verifier を順番に実行する。"""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    if request.refresh:
        await locked_official_sync(force=True, limit=max(1, min(request.limit, 100)), source="ask-refresh")
        return await ask_official_question(question, refresh=False, limit=request.limit)
    return await ask_official_question(question, refresh=False, limit=request.limit)


@app.delete("/answers/cache")
def delete_answers_cache():
    """保存済みの回答検証履歴を削除する。"""
    return {"deleted": delete_answer_cache()}


@app.get("/health")
async def health_check():
    """DBとAI接続の簡易ヘルスチェック。"""
    db_ok = DB_FILE.exists()
    models = []
    ai_status = "not_checked"
    if AI_PROVIDER == "ollama":
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(AI_BASE_URL.replace("/api/generate", "/api/tags"))
                models = [m["name"] for m in resp.json().get("models", [])]
            ai_status = "connected"
        except Exception as e:
            ai_status = f"error: {e}"
    elif AI_PROVIDER == "api":
        ai_status = (
            "configured"
            if AI_BASE_URL and AI_MODEL and AI_GENERATOR_MODEL and AI_VERIFIER_MODEL and AI_NORMALIZER_MODEL
            else "missing_config"
        )
        models = sorted({active_ai_model(), AI_GENERATOR_MODEL, AI_VERIFIER_MODEL, AI_NORMALIZER_MODEL})
    return {
        "status": "ok" if db_ok and ai_status in {"connected", "configured"} else "degraded",
        "db": "found" if db_ok else "missing",
        "ai_provider": AI_PROVIDER,
        "ai": ai_status,
        "ai_config": current_ai_config(),
        "models": models,
    }

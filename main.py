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
    DB_FILE,
    OFFICIAL_BACKGROUND_INTERVAL_MINUTES,
    OFFICIAL_HISTORY_PER_SOURCE,
)
from db import (
    count_simulated_events,
    delete_answer_cache,
    delete_simulated_events,
    get_db,
    load_latest_official_signals_by_source,
    load_official_history_for_source,
    load_safety_events,
    load_simulated_safety_events,
    save_area_official_signals,
)
from official_service import build_data_summary, live_official_summary, run_official_sync
from official_sources import fetch_jma_aichi_signals, official_status_catalog, source_catalog
from schemas import AskRequest
from simulated_events import build_simulated_events, scenario_catalog


app = FastAPI(title="Urban Safety Research Backend", version="4.0.0")
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
    return {"status": "ok", "version": "4.0.0", "theme": "公的情報と模擬イベントデータに基づく都市安全情報集約・可視化"}


@app.get("/system/overview")
def get_system_overview():
    """卒論説明にも使える、システム全体像とDB状態の概要。"""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM official_area_observations")
    official_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM answer_logs")
    answer_count = cur.fetchone()[0]
    simulated_count = count_simulated_events(conn)
    conn.close()
    return {
        "concept": "公的情報と模擬イベントデータに基づく都市安全情報集約・可視化システムの研究用プロトタイプ",
        "pipeline": [
            "公的情報源",
            "official/sync で取得",
            "LLMまたはルールで構造化",
            "研究用模擬イベントを必要に応じて追加",
            "都市安全情報DBへ保存",
            "モバイルUIで通常時・異常時を可視化",
            "ユーザー質問に対して保存済み情報から要約回答を生成",
            "参照情報・更新時刻・模擬データ有無を表示",
        ],
        "ai": current_ai_config(),
        "official_statuses": official_status_catalog(),
        "database": {
            "official_observation_count": official_count,
            "simulated_observation_count": simulated_count,
            "answer_log_count": answer_count,
        },
    }


@app.get("/dashboard")
def get_dashboard(include_simulated: bool = False):
    """フロント画面用に、最新都市安全情報・全体状態・AI設定を返す。"""
    conn = get_db()
    official_observations = load_latest_official_signals_by_source(conn, limit_per_source=1, include_simulated=False)
    simulated_observations = load_simulated_safety_events(conn, limit=20) if include_simulated else []
    conn.close()
    observations = simulated_observations + official_observations
    summary = live_official_summary(observations)
    return {
        "basis": "公的情報を基本に表示しています。模擬データを含める場合は研究検証用として明示します。",
        "data_summary": build_data_summary(observations),
        "official_summary": summary,
        "official_observations": observations,
        "official_count": len(observations),
        "include_simulated": include_simulated,
        "simulated_count": sum(1 for item in observations if item.get("is_simulated")),
        "ai_config": current_ai_config(),
        "display_policy": "evidence_with_simulation_label",
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
    include_simulated: bool = Query(default=False),
):
    """Evidence DB に保存済みの都市安全情報を返す。"""
    conn = get_db()
    if source:
        observations = load_official_history_for_source(conn, source, limit=limit)
    else:
        observations = load_latest_official_signals_by_source(
            conn,
            limit_per_source=1,
            include_simulated=include_simulated,
        )[:limit]
    conn.close()
    return {
        "source": source,
        "observations": observations,
        "count": len(observations),
        "summary": live_official_summary(observations),
        "include_simulated": include_simulated,
        "history_per_source": OFFICIAL_HISTORY_PER_SOURCE,
    }


@app.get("/safety/events")
def get_safety_events(
    limit: int = Query(default=50, ge=1, le=200),
    include_simulated: bool = Query(default=False),
    category: Optional[str] = Query(default=None),
    area: Optional[str] = Query(default=None),
    min_severity: Optional[float] = Query(default=None, ge=0),
):
    """統一データモデルで都市安全情報一覧を返す研究用API。"""
    conn = get_db()
    observations = load_safety_events(
        conn,
        limit=limit,
        include_simulated=include_simulated,
        category=category,
        area=area,
        min_severity=min_severity,
    )
    conn.close()
    return {
        "events": observations,
        "count": len(observations),
        "include_simulated": include_simulated,
        "filters": {
            "category": category,
            "area": area,
            "min_severity": min_severity,
        },
        "summary": live_official_summary(observations),
    }


@app.get("/safety/simulated-events/scenarios")
def get_simulated_event_scenarios():
    """研究検証用に利用できる模擬シナリオ一覧。"""
    return {
        "notice": "ここに含まれるデータは研究検証用であり、公的情報ではありません。",
        "scenarios": scenario_catalog(),
    }


@app.post("/safety/simulated-events/load")
async def load_simulated_event_scenario(
    scenario: str = Query(default="ollama_random"),
    mode: str = Query(default="replace", pattern="^(replace|append)$"),
    count: int = Query(default=20, ge=1, le=50),
    dangerous_ratio: float = Query(default=0.7, ge=0.0, le=1.0),
):
    """ローカルOllamaで生成した模擬シナリオを都市安全情報DBに保存する。"""
    try:
        events, generation = await build_simulated_events(
            scenario=scenario,
            count=count,
            dangerous_ratio=dangerous_ratio,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ollamaによる模擬データ生成に失敗しました: {type(exc).__name__}: {exc}") from exc
    conn = get_db()
    deleted = delete_simulated_events(conn) if mode == "replace" else 0
    saved = save_area_official_signals(conn, events)
    total_simulated = count_simulated_events(conn)
    conn.close()
    return {
        "scenario": scenario,
        "mode": mode,
        "deleted_simulated_events": deleted,
        "saved_simulated_events": saved,
        "total_simulated_events": total_simulated,
        "generation": generation,
        "notice": "保存されたデータは研究検証用の模擬データです。",
    }


@app.delete("/safety/simulated-events")
def clear_simulated_event_scenario():
    """Evidence DB から模擬イベントだけを削除する。"""
    conn = get_db()
    deleted = delete_simulated_events(conn)
    conn.close()
    return {"deleted_simulated_events": deleted}


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
    """ユーザー質問に対して、保存済み都市安全情報DBに基づく要約回答を返す。"""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    if request.refresh:
        await locked_official_sync(force=True, limit=max(1, min(request.limit, 100)), source="ask-refresh")
        return await ask_official_question(
            question,
            refresh=False,
            limit=request.limit,
            followup_context=request.followup_context,
            include_simulated=request.include_simulated,
            category=request.category,
            area=request.area,
            min_severity=request.min_severity,
        )
    return await ask_official_question(
        question,
        refresh=False,
        limit=request.limit,
        followup_context=request.followup_context,
        include_simulated=request.include_simulated,
        category=request.category,
        area=request.area,
        min_severity=request.min_severity,
    )


@app.delete("/answers/cache")
def delete_answers_cache():
    """保存済みの問い合わせ回答ログを削除する。"""
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
            if AI_BASE_URL and AI_MODEL and AI_GENERATOR_MODEL and AI_NORMALIZER_MODEL
            else "missing_config"
        )
        models = sorted({active_ai_model(), AI_GENERATOR_MODEL, AI_NORMALIZER_MODEL})
    return {
        "status": "ok" if db_ok and ai_status in {"connected", "configured"} else "degraded",
        "db": "found" if db_ok else "missing",
        "ai_provider": AI_PROVIDER,
        "ai": ai_status,
        "ai_config": current_ai_config(),
        "models": models,
    }

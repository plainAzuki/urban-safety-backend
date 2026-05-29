"""Ollama で研究検証用の模擬イベントデータを生成する。

模擬イベントは公的情報ではない。保存時は必ず is_simulated=True とし、
UI/API/レポートで「模擬データ」と明示する。
"""

import json
from datetime import datetime, timedelta
from typing import Optional

from ai_client import call_ollama
from config import AI_MODEL
from json_utils import extract_json_object


SIMULATED_SOURCE = "研究用Ollama模擬イベントデータ"
SIMULATED_SOURCE_URL = "simulation://urban-safety-research/ollama"
DEFAULT_EVENT_COUNT = 20
DEFAULT_DANGEROUS_RATIO = 0.7

VALID_CATEGORIES = {"鉄道", "道路", "気象", "防災", "空港", "ライフライン", "その他"}
VALID_STATUSES = {"通常", "情報", "注意", "警戒", "危険", "運休", "支障"}
SAFE_STATUSES = {"通常", "情報"}
DANGEROUS_STATUSES = {"注意", "警戒", "危険", "運休", "支障"}


SCENARIOS = {
    "ollama_random": {
        "name": "Ollamaランダム生成",
        "description": "ローカル Ollama が20件の都市安全模擬イベントをJSON形式で生成する。",
    },
    "natural_disaster": {
        "name": "自然災害中心",
        "description": "大雨・暴風・大雪・高潮・地震・停電などを中心に生成する。",
    },
    "transport_disruption": {
        "name": "交通障害中心",
        "description": "鉄道運転見合わせ、道路通行止め、空港アクセス注意などを中心に生成する。",
    },
    "mixed": {
        "name": "複合シナリオ",
        "description": "自然災害、停運、道路規制、避難情報を混在させて生成する。",
    },
    "multi_event": {
        "name": "複合シナリオ",
        "description": "既存フロントエンド互換の名称。Ollamaで自然災害、停運、道路規制、避難情報を混在生成する。",
    },
}


def now_text() -> str:
    """模擬イベントの発生・更新時刻を統一形式で返す。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def scenario_catalog() -> list[dict]:
    """APIで返す模擬シナリオ一覧。"""
    return [{"key": key, **value} for key, value in SCENARIOS.items()]


def build_generation_prompt(
    scenario: str = "ollama_random",
    count: int = DEFAULT_EVENT_COUNT,
    dangerous_ratio: float = DEFAULT_DANGEROUS_RATIO,
) -> str:
    """Ollama に渡す固定JSON形式の生成プロンプト。"""
    dangerous_count = round(count * dangerous_ratio)
    safe_count = count - dangerous_count
    scenario_info = SCENARIOS.get(scenario, SCENARIOS["ollama_random"])
    return f"""あなたは卒業研究用の都市安全情報シミュレータです。
愛知県および周辺地域を対象に、研究検証用の模擬イベントデータを生成してください。

重要:
- 実際の公的情報ではなく、すべて研究用の模擬データです。
- 必ずJSON objectのみを返してください。Markdownや説明文は禁止です。
- events は必ず {count} 件にしてください。
- 低リスク・無危険データを約3割、危険・支障ありデータを約7割にしてください。
- 今回は低リスク {safe_count} 件、高リスク {dangerous_count} 件を目安にしてください。
- 自然災害、鉄道停運、道路通行止め、気象警報、避難情報、空港アクセス注意、停電・断水などを含めてください。
- 同じ地域・同じタイトルだけに偏らず、名古屋市、豊橋市、岡崎市、常滑市、知多半島、三河地方、愛知県西部などを混ぜてください。
- 低リスクイベントの status は 通常 または 情報 にしてください。
- 高リスクイベントの status は 注意, 警戒, 危険, 運休, 支障 のいずれかにしてください。
- severity は 0.0 から 5.0 の数値にしてください。低リスクは 0.0〜1.5、高リスクは 2.0〜5.0 を目安にしてください。
- category は 鉄道, 道路, 気象, 防災, 空港, ライフライン, その他 のいずれかにしてください。
- source_url は空文字で構いません。
- observed_at と updated_at は空文字で構いません。サーバー側で現在時刻に補完します。

シナリオ:
- key: {scenario}
- name: {scenario_info["name"]}
- description: {scenario_info["description"]}

返却JSON schema:
{{
  "events": [
    {{
      "category": "鉄道",
      "area": "名古屋市",
      "label": "短いタイトル",
      "status": "支障",
      "severity": 3.0,
      "detail": "利用者が状況を理解できる1文から2文の説明",
      "observed_at": "",
      "updated_at": "",
      "source_url": ""
    }}
  ]
}}
"""


async def build_simulated_events(
    scenario: str = "ollama_random",
    count: int = DEFAULT_EVENT_COUNT,
    dangerous_ratio: float = DEFAULT_DANGEROUS_RATIO,
    model: Optional[str] = None,
) -> tuple[list[dict], dict]:
    """ローカル Ollama で指定件数の模擬イベントを生成する。"""
    if scenario not in SCENARIOS:
        valid = ", ".join(SCENARIOS)
        raise ValueError(f"unknown scenario: {scenario}. valid scenarios: {valid}")
    safe_count = max(0, min(count, count - round(count * dangerous_ratio)))
    dangerous_count = count - safe_count
    prompt = build_generation_prompt(scenario=scenario, count=count, dangerous_ratio=dangerous_ratio)
    raw_output = await call_ollama(prompt, json_mode=True, model=model or AI_MODEL, num_predict=6000)
    data = extract_json_object(raw_output)
    raw_events = data.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("Ollama出力に events 配列がありません。")
    cleaned = clean_generated_events(raw_events, scenario=scenario, count=count)
    cleaned = enforce_risk_ratio(cleaned, safe_count=safe_count, dangerous_count=dangerous_count)
    cleaned = interleave_risk_levels(cleaned)
    stamp_simulated_event_order(cleaned, scenario=scenario)
    metadata = {
        "generator": "local_ollama",
        "model": model or AI_MODEL,
        "scenario": scenario,
        "requested_count": count,
        "saved_count": len(cleaned),
        "target_safe_count": safe_count,
        "target_dangerous_count": dangerous_count,
    }
    return cleaned, metadata


def clean_generated_events(raw_events: list[dict], scenario: str, count: int) -> list[dict]:
    """Ollama のJSONをDB保存用の固定形式へ丸める。"""
    events = []
    timestamp = now_text()
    for index, raw in enumerate(raw_events[:count], start=1):
        if not isinstance(raw, dict):
            continue
        category = clean_choice(raw.get("category"), VALID_CATEGORIES, "その他")
        status = clean_choice(raw.get("status"), VALID_STATUSES, "情報")
        severity = clean_severity(raw.get("severity"), status)
        label = clean_text(raw.get("label"), f"模擬イベント{index}", 120)
        area = clean_text(raw.get("area"), "愛知県", 80)
        detail = clean_text(raw.get("detail"), "研究検証用に生成された都市安全情報です。", 900)
        observed_at = clean_text(raw.get("observed_at"), "", 40) or timestamp
        updated_at = clean_text(raw.get("updated_at"), "", 40) or timestamp
        event_id = f"simulation:{scenario}:ollama:{timestamp}:{index:02d}"
        events.append({
            "id": event_id,
            "source": SIMULATED_SOURCE,
            "source_url": clean_text(raw.get("source_url"), "", 500) or f"{SIMULATED_SOURCE_URL}/{scenario}/{index:02d}",
            "category": category,
            "area": area,
            "label": label,
            "display_label": f"【模擬データ】{label}",
            "severity": severity,
            "status": status,
            "detail": f"【模擬データ】{detail}",
            "observed_at": observed_at,
            "updated_at": updated_at,
            "is_simulated": True,
        })
    if len(events) != count:
        raise ValueError(f"Ollama生成件数が不足しています: expected={count}, actual={len(events)}")
    return events


def enforce_risk_ratio(events: list[dict], safe_count: int, dangerous_count: int) -> list[dict]:
    """3:7 に近い低リスク・高リスク比率へ補正する。"""
    safe_indexes = [index for index, item in enumerate(events) if item["status"] in SAFE_STATUSES]
    dangerous_indexes = [index for index, item in enumerate(events) if item["status"] in DANGEROUS_STATUSES]

    while len(safe_indexes) < safe_count and dangerous_indexes:
        index = dangerous_indexes.pop()
        events[index]["status"] = "情報"
        events[index]["severity"] = min(float(events[index]["severity"]), 1.5)
        events[index]["label"] = f"{events[index]['label']}（状況確認）"
        events[index]["display_label"] = f"【模擬データ】{events[index]['label']}"
        safe_indexes.append(index)

    safe_indexes = [index for index, item in enumerate(events) if item["status"] in SAFE_STATUSES]
    while len(safe_indexes) > safe_count:
        index = safe_indexes.pop()
        events[index]["status"] = "注意"
        events[index]["severity"] = max(float(events[index]["severity"]), 2.0)

    final_safe = sum(1 for item in events if item["status"] in SAFE_STATUSES)
    final_dangerous = sum(1 for item in events if item["status"] in DANGEROUS_STATUSES)
    if final_safe != safe_count or final_dangerous != dangerous_count:
        raise ValueError(f"危険比率の補正に失敗しました: safe={final_safe}, dangerous={final_dangerous}")
    return events


def interleave_risk_levels(events: list[dict]) -> list[dict]:
    """一覧表示で低リスクと高リスクが偏らないように並べ替える。"""
    safe_events = [item for item in events if item["status"] in SAFE_STATUSES]
    dangerous_events = [item for item in events if item["status"] in DANGEROUS_STATUSES]
    mixed = []
    while safe_events or dangerous_events:
        for _ in range(2):
            if dangerous_events:
                mixed.append(dangerous_events.pop(0))
        if safe_events:
            mixed.append(safe_events.pop(0))
    return mixed


def stamp_simulated_event_order(events: list[dict], scenario: str) -> None:
    """DB取得時にも生成順を再現できるよう、IDと時刻に順序を埋め込む。"""
    base_time = datetime.now()
    for index, event in enumerate(events, start=1):
        timestamp = (base_time + timedelta(seconds=index)).strftime("%Y-%m-%d %H:%M:%S")
        event["id"] = f"simulation:{scenario}:ollama:{base_time.strftime('%Y%m%d%H%M%S')}:{index:02d}"
        event["source_url"] = f"{SIMULATED_SOURCE_URL}/{scenario}/{index:02d}"
        event["observed_at"] = timestamp
        event["updated_at"] = timestamp


def clean_choice(value: object, allowed: set[str], fallback: str) -> str:
    """許可値以外をフォールバックへ丸める。"""
    text = str(value or "").strip()
    return text if text in allowed else fallback


def clean_text(value: object, fallback: str, max_chars: int) -> str:
    """表示用文字列を安全な長さへ丸める。"""
    text = str(value or "").strip()
    return (text or fallback)[:max_chars]


def clean_severity(value: object, status: str) -> float:
    """重要度を0.0から5.0へ丸め、status と大きく矛盾しない値にする。"""
    try:
        severity = float(value)
    except (TypeError, ValueError):
        severity = 0.5 if status in SAFE_STATUSES else 3.0
    severity = max(0.0, min(5.0, severity))
    if status in SAFE_STATUSES:
        return min(severity, 1.5)
    return max(severity, 2.0)

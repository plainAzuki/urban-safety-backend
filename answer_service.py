"""保存済み都市安全情報に基づく自然言語問い合わせ回答。"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from ai_client import call_ai
from config import AI_GENERATOR_MODEL, AI_PROVIDER
from db import get_db, load_safety_events, save_answer_log
from official_service import run_official_sync
from prompts import build_answer_prompt


LATEST_ASK_REPORT_FILE = Path(__file__).parent / "latest_ask_report.md"


def json_block(data) -> str:
    """Markdown report用にJSONを読みやすく整形する。"""
    return json.dumps(data, ensure_ascii=False, indent=2)


def build_ask_report(
    answer_id: str,
    question: str,
    followup_context: str,
    db_observations: list[dict],
    prompt_observations: list[dict],
    generator_prompt: str,
    answer: str,
    references: list[dict],
    includes_simulated: bool,
    generator_model: str,
    provider: str,
    ai_error: Optional[str],
) -> str:
    """1回の問い合わせに対する研究・デバッグ用レポートを作る。"""
    return f"""# Ask Report

generated_at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
answer_id: {answer_id}
provider: {provider}
generator_model: {generator_model}
research_policy: 保存済み都市安全情報DBに基づく要約回答

## Question

{question}

## Follow-up Context

{followup_context or "(none)"}

## Result Summary

- response_type: EVIDENCE_SUMMARY
- answer_shown_to_user: True
- references_count: {len(references)}
- includes_simulated: {includes_simulated}
- ai_error: {ai_error or ""}

## Answer

{answer}

## References

```json
{json_block(references)}
```

## Evidence DB Used By Prompt

```json
{json_block(prompt_observations)}
```

## Full Evidence DB Snapshot

```json
{json_block(db_observations)}
```

## Generator Prompt

```text
{generator_prompt}
```
"""


def save_latest_ask_report(report: str) -> None:
    """最新の質問レポートだけを上書き保存する。"""
    LATEST_ASK_REPORT_FILE.write_text(report, encoding="utf-8")


async def ask_official_question(
    question: str,
    refresh: bool = False,
    limit: int = 20,
    followup_context: str = "",
    include_simulated: bool = False,
    category: Optional[str] = None,
    area: Optional[str] = None,
    min_severity: Optional[float] = None,
) -> dict:
    """保存済み都市安全情報DBに基づく要約回答を返す。"""
    if refresh:
        await run_official_sync(force=True, limit=max(1, min(limit, 100)), source="ask-refresh")
    conn = get_db()
    observations = select_relevant_observations(
        question,
        load_safety_events(
            conn,
            limit=max(20, min(limit, 100)),
            include_simulated=include_simulated,
            simulated_only=include_simulated,
            category=category or None,
            area=area or None,
            min_severity=min_severity,
        ),
        limit=limit,
    )
    db_observations = load_safety_events(
        conn,
        limit=1000,
        include_simulated=include_simulated,
        simulated_only=include_simulated,
    )
    conn.close()

    ai_error = None
    generator_prompt = build_answer_prompt(question, observations, followup_context=followup_context)
    try:
        answer, model = await call_ai(generator_prompt, model=AI_GENERATOR_MODEL)
    except Exception as exc:
        answer = build_fallback_answer(question, observations)
        model = "answer-generation-fallback"
        ai_error = str(exc)

    references = build_references(observations)
    includes_simulated = any(item.get("is_simulated") for item in observations)
    answer_id = save_answer_log(
        question=question,
        answer=answer,
        references=references,
        includes_simulated=includes_simulated,
        model=model,
        provider=AI_PROVIDER,
        ai_error=ai_error,
    )
    report = build_ask_report(
        answer_id=answer_id,
        question=question,
        followup_context=followup_context,
        db_observations=db_observations,
        prompt_observations=observations,
        generator_prompt=generator_prompt,
        answer=answer,
        references=references,
        includes_simulated=includes_simulated,
        generator_model=model,
        provider=AI_PROVIDER,
        ai_error=ai_error,
    )
    save_latest_ask_report(report)
    return {
        "id": answer_id,
        "question": question,
        "draft_answer": answer,
        "answer": answer,
        "generation_policy": "保存済み都市安全情報DBに基づく要約回答",
        "references": references,
        "includes_simulated": includes_simulated,
        "simulated_reference_count": sum(1 for item in observations if item.get("is_simulated")),
        "model": model,
        "provider": AI_PROVIDER,
        "ai_error": ai_error,
        "official_context_count": len(observations),
        "report_path": str(LATEST_ASK_REPORT_FILE),
    }


def select_relevant_observations(question: str, observations: list[dict], limit: int = 20) -> list[dict]:
    """質問語に近い情報を優先し、リスクの高い情報も残す。"""
    keywords = [word for word in re_split_question(question) if len(word) >= 2]

    def score(item: dict) -> tuple[float, str]:
        text = " ".join(str(item.get(key) or "") for key in ("source", "category", "area", "label", "detail", "status"))
        keyword_score = sum(1 for keyword in keywords if keyword in text)
        severity = float(item.get("severity") or 0)
        simulated_penalty = 0.0 if item.get("is_simulated") else 0.2
        return keyword_score * 10 + severity + simulated_penalty, str(item.get("updated_at") or item.get("observed_at") or "")

    ranked = sorted(observations, key=score, reverse=True)
    return ranked[: max(1, min(limit, 50))]


def re_split_question(question: str) -> list[str]:
    """日本語・英数字混在の質問から簡易キーワードを取り出す。"""
    import re

    tokens = re.split(r"[\s、。,.!?！？/・（）()]+", question)
    return [token.strip() for token in tokens if token.strip()]


def build_references(observations: list[dict]) -> list[dict]:
    """フロントエンドと実験記録で使う参照情報を作る。"""
    return [
        {
            "source": item.get("source"),
            "category": item.get("category"),
            "area": item.get("area"),
            "label": item.get("display_label") or item.get("label"),
            "status": item.get("status"),
            "severity": item.get("severity"),
            "updated_at": item.get("updated_at") or item.get("created_at") or item.get("observed_at"),
            "source_url": item.get("source_url"),
            "is_simulated": bool(item.get("is_simulated")),
        }
        for item in observations
    ]


def build_fallback_answer(question: str, observations: list[dict]) -> str:
    """AIが使えない場合でも研究デモを継続できる決定的な要約を返す。"""
    if not observations:
        return "保存済みの都市安全情報がないため、この質問に回答できません。公的情報の更新、または模擬データの生成を行ってから再確認してください。"
    lines = [f"質問「{question}」に関連する保存済み情報の要約です。"]
    for item in observations[:5]:
        simulated = "模擬データ" if item.get("is_simulated") else "公的情報"
        updated = item.get("updated_at") or item.get("created_at") or item.get("observed_at")
        lines.append(
            f"{item.get('category', 'その他')}・{item.get('area')}: {item.get('display_label') or item.get('label')} "
            f"（状態: {item.get('status')}、{simulated}、更新: {updated}）。"
        )
    if any(item.get("is_simulated") for item in observations):
        lines.append("上記には研究検証用の模擬データが含まれます。")
    return "\n".join(lines)

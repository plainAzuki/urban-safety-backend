"""ユーザー質問への都市安全情報要約。

Verifier Agent は本研究の主題から外し、保存済みの公的情報・模擬イベントを
明示的な参照情報として返す方針にしている。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from ai_client import call_ai
from config import AI_GENERATOR_MODEL, AI_PROVIDER, AI_VERIFIER_MODEL
from db import get_db, load_area_official_signals, load_safety_events, save_answer_verification
from json_utils import extract_json_object
from official_service import run_official_sync
from prompts import build_answer_prompt, build_verifier_prompt


LATEST_ASK_REPORT_FILE = Path(__file__).parent / "latest_ask_report.md"


def clean_verification(data: dict) -> dict:
    """Verifier のJSON出力をアプリ内部の固定形式へ正規化する。"""
    verdict = str(data.get("verdict") or "NEEDS_REVIEW").strip()
    if verdict not in {"PASS", "FAIL", "NEEDS_REVIEW"}:
        verdict = "NEEDS_REVIEW"
    expected_policy = {
        "PASS": "SHOW",
        "NEEDS_REVIEW": "SHOW_WITH_WARNING",
        "FAIL": "DO_NOT_SHOW",
    }[verdict]
    display_policy = str(data.get("display_policy") or expected_policy).strip()
    if display_policy not in {"SHOW", "SHOW_WITH_WARNING", "DO_NOT_SHOW"}:
        display_policy = expected_policy
    if verdict == "FAIL":
        display_policy = "DO_NOT_SHOW"
    reasons = data.get("reasons")
    if not isinstance(reasons, list):
        reasons = [str(reasons)] if reasons else []
    checked_claims = data.get("checked_claims")
    if not isinstance(checked_claims, list):
        checked_claims = []
    return {
        "verdict": verdict,
        "display_policy": display_policy,
        "warning": str(data.get("warning") or "").strip(),
        "reasons": [str(item)[:300] for item in reasons[:6]],
        "checked_claims": [str(item)[:300] for item in checked_claims[:8]],
    }


async def verify_answer(
    question: str,
    draft_answer: str,
    observations: list[dict],
    verifier_prompt: str,
) -> tuple[dict, Optional[str], str]:
    """draft answer を公式情報と照合し、表示可否を決める。"""
    try:
        output, _ = await call_ai(
            verifier_prompt,
            json_mode=True,
            model=AI_VERIFIER_MODEL,
        )
        return clean_verification(extract_json_object(output)), None, output
    except Exception as exc:
        return {
            "verdict": "NEEDS_REVIEW",
            "display_policy": "SHOW_WITH_WARNING",
            "warning": "Verifier Agent の確認に失敗したため、注意付きで表示します。",
            "reasons": [f"{type(exc).__name__}: {exc}"],
            "checked_claims": [],
        }, str(exc), ""


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
    draft_answer: str,
    verifier_prompt: str,
    verifier_raw_output: str,
    verification: dict,
    visible_answer: Optional[str],
    generator_model: str,
    provider: str,
    ai_error: Optional[str],
) -> str:
    """1回の質問に対する研究・デバッグ用レポートを作る。"""
    shown = visible_answer is not None
    return f"""# Ask Report

generated_at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
answer_id: {answer_id}
provider: {provider}
generator_model: {generator_model}
research_policy: Evidence DB に基づく要約。Verifier Agent は研究主題から除外。

## Question

{question}

## Follow-up Context

{followup_context or "(none)"}

## Result Summary

- response_type: {verification.get("verdict")}
- display_policy: {verification.get("display_policy")}
- answer_shown_to_user: {shown}
- ai_error: {ai_error or ""}

## Visible Answer

{visible_answer or "(no answer)"}

## Draft Answer

{draft_answer}

## Verification Policy

```json
{json_block(verification)}
```

## Verifier Agent

```text
研究方針により、このレポートでは Verifier Agent を使用していません。
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

## Verifier Prompt

```text
{verifier_prompt}
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
    """Evidence DB に基づく要約回答を返す。Verifier Agent は使用しない。"""
    if refresh:
        await run_official_sync(force=True, limit=max(1, min(limit, 100)), source="ask-refresh")
    conn = get_db()
    observations = select_relevant_observations(
        question,
        load_safety_events(
            conn,
            limit=max(20, min(limit, 100)),
            include_simulated=include_simulated,
            category=category or None,
            area=area or None,
            min_severity=min_severity,
        ),
        limit=limit,
    )
    db_observations = load_area_official_signals(conn, limit=1000, include_simulated=include_simulated)
    conn.close()

    ai_error = None
    generator_prompt = build_answer_prompt(question, observations, followup_context=followup_context)
    try:
        draft_answer, model = await call_ai(generator_prompt, model=AI_GENERATOR_MODEL)
    except Exception as exc:
        draft_answer = build_fallback_answer(question, observations)
        model = "answer-generation-fallback"
        ai_error = str(exc)

    verifier_prompt = "Verifier Agent は本研究テーマから外しているため使用しません。"
    verifier_raw_output = ""
    verification = {
        "verdict": "REFERENCE_SUMMARY",
        "display_policy": "SHOW",
        "warning": "この回答は保存済みの都市安全情報に基づく要約です。模擬データを含む場合は実際の公的発表ではありません。",
        "reasons": ["研究方針により Verifier Agent は使用せず、参照情報と模擬データ有無を明示します。"],
        "checked_claims": [],
    }

    visible_answer = draft_answer
    answer_id = save_answer_verification(
        question,
        draft_answer,
        visible_answer,
        verification,
        model,
        AI_PROVIDER,
        ai_error,
    )
    report = build_ask_report(
        answer_id=answer_id,
        question=question,
        followup_context=followup_context,
        db_observations=db_observations,
        prompt_observations=observations,
        generator_prompt=generator_prompt,
        draft_answer=draft_answer,
        verifier_prompt=verifier_prompt,
        verifier_raw_output=verifier_raw_output,
        verification=verification,
        visible_answer=visible_answer,
        generator_model=model,
        provider=AI_PROVIDER,
        ai_error=ai_error,
    )
    save_latest_ask_report(report)
    references = build_references(observations)
    return {
        "id": answer_id,
        "question": question,
        "draft_answer": draft_answer,
        "answer": visible_answer,
        "verification": verification,
        "generation_policy": "保存済み都市安全情報に基づく要約。Verifier Agent は使用しない。",
        "references": references,
        "includes_simulated": any(item.get("is_simulated") for item in observations),
        "simulated_reference_count": sum(1 for item in observations if item.get("is_simulated")),
        "model": model,
        "provider": AI_PROVIDER,
        "ai_error": ai_error,
        "official_context_count": len(observations),
        "report_path": str(LATEST_ASK_REPORT_FILE),
    }


def select_relevant_observations(question: str, observations: list[dict], limit: int = 20) -> list[dict]:
    """質問語に近い情報を優先し、重要度の高い情報も残す。"""
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
        return "保存済みの都市安全情報がないため、この質問に回答できません。公式情報を更新してから再確認してください。"
    lines = [f"質問「{question}」に関連する保存済み情報の要約です。"]
    for item in observations[:5]:
        simulated = "模擬データ" if item.get("is_simulated") else "公的情報"
        updated = item.get("updated_at") or item.get("created_at") or item.get("observed_at")
        lines.append(
            f"{item.get('category', 'その他')}・{item.get('area')}: {item.get('display_label') or item.get('label')} "
            f"（状態: {item.get('status')}、重要度: {item.get('severity')}、{simulated}、更新: {updated}）。"
        )
    if any(item.get("is_simulated") for item in observations):
        lines.append("上記には研究検証用の模擬データが含まれます。実際の判断には公的機関の最新情報を確認してください。")
    return "\n".join(lines)

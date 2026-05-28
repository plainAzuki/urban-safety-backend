"""ユーザー質問への回答生成と Verifier Agent の判定。"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from ai_client import call_ai
from config import AI_GENERATOR_MODEL, AI_PROVIDER, AI_VERIFIER_MODEL
from db import get_db, load_area_official_signals, load_latest_official_signals_by_source, save_answer_verification
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
    verifier_passed = verification.get("verdict") == "PASS"
    shown = visible_answer is not None
    return f"""# Ask Report

generated_at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
answer_id: {answer_id}
provider: {provider}
generator_model: {generator_model}
verifier_model: {AI_VERIFIER_MODEL}

## Question

{question}

## Follow-up Context

{followup_context or "(none)"}

## Result Summary

- verifier_passed: {verifier_passed}
- verdict: {verification.get("verdict")}
- display_policy: {verification.get("display_policy")}
- answer_shown_to_user: {shown}
- ai_error: {ai_error or ""}

## Visible Answer

{visible_answer or "(blocked by Verifier Agent)"}

## Draft Answer

{draft_answer}

## Verifier Analysis

```json
{json_block(verification)}
```

## Verifier Raw Output

```text
{verifier_raw_output or "(no raw output)"}
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
) -> dict:
    """Generator → Verifier → 保存 → API返却値作成の一連の流れ。"""
    if refresh:
        await run_official_sync(force=True, limit=max(1, min(limit, 100)), source="ask-refresh")
    conn = get_db()
    observations = load_latest_official_signals_by_source(conn, limit_per_source=1)
    db_observations = load_area_official_signals(conn, limit=1000)
    conn.close()

    ai_error = None
    generator_prompt = build_answer_prompt(question, observations, followup_context=followup_context)
    try:
        draft_answer, model = await call_ai(generator_prompt, model=AI_GENERATOR_MODEL)
    except Exception as exc:
        draft_answer = "AI回答生成に失敗しました。公式情報一覧を確認してください。"
        model = "answer-generation-fallback"
        ai_error = str(exc)

    verifier_prompt = build_verifier_prompt(question, draft_answer, observations)
    verification, verifier_error, verifier_raw_output = await verify_answer(
        question,
        draft_answer,
        observations,
        verifier_prompt,
    )
    if verifier_error:
        ai_error = f"{ai_error or ''} verifier: {verifier_error}".strip()

    visible_answer = draft_answer if verification["display_policy"] != "DO_NOT_SHOW" else None
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
    return {
        "id": answer_id,
        "question": question,
        "draft_answer": draft_answer,
        "answer": visible_answer,
        "verification": verification,
        "model": model,
        "provider": AI_PROVIDER,
        "ai_error": ai_error,
        "official_context_count": len(observations),
        "report_path": str(LATEST_ASK_REPORT_FILE),
    }

"""LLM出力JSONを扱う共通ユーティリティ。"""

import json
import re


def extract_json_object(text: str) -> dict:
    """Markdown混じりのLLM出力からJSON objectを取り出す。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))

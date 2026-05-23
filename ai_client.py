"""LLM 呼び出しを隠蔽する薄いクライアント層。"""

from typing import Optional

import httpx

from config import (
    AI_API_KEY,
    AI_BASE_URL,
    AI_GENERATOR_MODEL,
    AI_MODEL,
    AI_NORMALIZER_MODEL,
    AI_PROVIDER,
    AI_THINK,
    AI_TIMEOUT_SECONDS,
    AI_VERIFIER_MODEL,
)


def active_ai_model() -> str:
    """UIやログに表示する現在のモデル名。"""
    return AI_MODEL or "未設定"


def current_ai_config() -> dict:
    """フロントエンドへ返してよい範囲のAI設定。"""
    config = {
        "provider": AI_PROVIDER,
        "model": active_ai_model(),
        "generator_model": AI_GENERATOR_MODEL or active_ai_model(),
        "verifier_model": AI_VERIFIER_MODEL or active_ai_model(),
        "normalizer_model": AI_NORMALIZER_MODEL or active_ai_model(),
        "base_url": AI_BASE_URL,
        "json_mode": AI_PROVIDER == "ollama",
    }
    if AI_THINK:
        config["think"] = AI_THINK
    if AI_PROVIDER == "api":
        config["api_key_ready"] = bool(AI_API_KEY)
    return config


def require_ai_config(model: Optional[str] = None) -> None:
    """LLM呼び出し前に最低限の設定不備を検出する。"""
    if AI_PROVIDER not in {"ollama", "api"}:
        raise RuntimeError(f"未対応のAI_PROVIDERです: {AI_PROVIDER}")
    if not model and not AI_MODEL:
        raise RuntimeError("AI_MODEL が設定されていません。")
    if not AI_BASE_URL:
        raise RuntimeError("AI_BASE_URL が設定されていません。")


async def call_ollama(prompt: str, json_mode: bool = False, model: Optional[str] = None) -> str:
    """Ollama の generate API を呼び出す。"""
    active_model = model or AI_MODEL
    require_ai_config(active_model)
    payload = {
        "model": active_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1 if json_mode else 0.2, "num_predict": 1200 if json_mode else 700},
    }
    if json_mode:
        payload["format"] = "json"
    if AI_THINK in {"true", "false"}:
        payload["think"] = AI_THINK == "true"
    async with httpx.AsyncClient(timeout=AI_TIMEOUT_SECONDS) as client:
        resp = await client.post(AI_BASE_URL, json=payload)
        resp.raise_for_status()
        return resp.json()["response"].strip()


async def call_api(prompt: str, json_mode: bool = False, model: Optional[str] = None) -> str:
    """OpenAI互換APIまたはResponses風APIを呼び出す。"""
    active_model = model or AI_MODEL
    require_ai_config(active_model)
    if AI_BASE_URL.endswith("/chat/completions"):
        payload = {
            "model": active_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1 if json_mode else 0.2,
            "max_tokens": 1200 if json_mode else 700,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
    else:
        payload = {
            "model": active_model,
            "input": prompt,
            "temperature": 0.1 if json_mode else 0.2,
            "max_output_tokens": 1200 if json_mode else 700,
        }
    headers = {"Authorization": f"Bearer {AI_API_KEY}"} if AI_API_KEY else {}
    async with httpx.AsyncClient(timeout=AI_TIMEOUT_SECONDS) as client:
        resp = await client.post(AI_BASE_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if data.get("output_text"):
            return data["output_text"].strip()
        if data.get("response"):
            return str(data["response"]).strip()
        if data.get("text"):
            return str(data["text"]).strip()
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content") or choices[0].get("text")
            if content:
                return str(content).strip()
        texts = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    texts.append(content.get("text", ""))
        return "\n".join(texts).strip()


async def call_ai(prompt: str, json_mode: bool = False, model: Optional[str] = None) -> tuple[str, str]:
    """プロバイダ差を吸収して、本文とモデル名を返す。"""
    active_model = model or AI_MODEL
    if AI_PROVIDER == "ollama":
        return await call_ollama(prompt, json_mode=json_mode, model=active_model), active_model
    if AI_PROVIDER == "api":
        return await call_api(prompt, json_mode=json_mode, model=active_model), active_model
    raise RuntimeError(f"未対応のAI_PROVIDERです: {AI_PROVIDER}")

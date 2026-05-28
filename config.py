"""アプリ設定値。"""

import os
from pathlib import Path


def env_value(name: str, default: str = "") -> str:
    """空文字を設定値として扱わないための環境変数ヘルパー。"""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


# AI実行環境のデフォルト値の設定。
AI_PROVIDER = env_value("AI_PROVIDER", "ollama").lower()
AI_MODEL = env_value("AI_MODEL", "qwen3.6:35b-a3b")
AI_GENERATOR_MODEL = env_value("AI_GENERATOR_MODEL", AI_MODEL)
AI_VERIFIER_MODEL = env_value("AI_VERIFIER_MODEL", AI_MODEL)
AI_NORMALIZER_MODEL = env_value("AI_NORMALIZER_MODEL", AI_MODEL)
AI_BASE_URL = env_value(
    "AI_BASE_URL",
    "http://localhost:11434/api/generate" if AI_PROVIDER == "ollama" else "",
)
AI_API_KEY = env_value("AI_API_KEY")
AI_THINK = env_value("AI_THINK", "false").lower()
AI_TIMEOUT_SECONDS = float(env_value("AI_TIMEOUT_SECONDS", "240"))

# 公式情報の構造化と保持に関する設定。
OFFICIAL_FETCH_MIN_INTERVAL_MINUTES = int(env_value("OFFICIAL_FETCH_MIN_INTERVAL_MINUTES", "60"))
OFFICIAL_LLM_BATCH_SIZE = int(env_value("OFFICIAL_LLM_BATCH_SIZE", "2"))
OFFICIAL_LLM_RAW_CHARS = int(env_value("OFFICIAL_LLM_RAW_CHARS", "1200"))
OFFICIAL_BACKGROUND_INTERVAL_MINUTES = int(env_value("OFFICIAL_BACKGROUND_INTERVAL_MINUTES", "1"))
OFFICIAL_HISTORY_PER_SOURCE = int(env_value("OFFICIAL_HISTORY_PER_SOURCE", "5"))

# SQLite DB はバックエンドディレクトリ直下に固定する。
DB_FILE = Path(env_value("DB_FILE", str(Path(__file__).parent / "urban_safety.db")))

"""FastAPI の入出力モデル。"""

from typing import Optional

from pydantic import BaseModel


class AskRequest(BaseModel):
    """ユーザー質問用のリクエスト。"""

    question: str
    followup_context: str = ""
    refresh: bool = False
    limit: int = 20
    include_simulated: bool = False
    category: Optional[str] = None
    area: Optional[str] = None
    min_severity: Optional[float] = None

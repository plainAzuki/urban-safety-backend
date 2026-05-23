"""FastAPI の入出力モデル。"""

from pydantic import BaseModel


class AskRequest(BaseModel):
    """ユーザー質問用のリクエスト。"""

    question: str
    refresh: bool = False
    limit: int = 20

"""Generator / Verifier / 公式情報正規化のプロンプト。"""

import json

from config import OFFICIAL_LLM_RAW_CHARS


def official_context_text(observations: list[dict]) -> str:
    """Verifier と Generator に渡す Evidence DB の短いテキスト表現。"""
    if not observations:
        return "取得済みの公式情報はありません。"
    lines = []
    for item in observations[:20]:
        lines.append(
            f"- {item.get('source')} / {item.get('area')} / {item.get('label')} / "
            f"status={item.get('status')} / "
            f"observed_at={item.get('observed_at')} / url={item.get('source_url')} / detail={item.get('detail')}"
        )
    return "\n".join(lines)


def build_official_normalization_prompt(records: list[dict]) -> str:
    """公式サイトの原文を Evidence DB の共通形式へ変換するためのプロンプト。"""
    compact_records = []
    for record in records:
        compact_records.append({
            "source": record.get("source"),
            "source_url": record.get("source_url") or record.get("url"),
            "area": record.get("area"),
            "category": record.get("category"),
            "url": record.get("url"),
            "title": record.get("title"),
            "observed_at": record.get("observed_at"),
            "raw_text": (record.get("raw_text") or "")[:OFFICIAL_LLM_RAW_CHARS],
        })
    raw_json = json.dumps(compact_records, ensure_ascii=False, indent=2)
    return f"""あなたは公式防災・交通情報を構造化するアナリストです。
以下の「公式サイト/APIから取得した原文データ」だけを読み、アプリが扱う信号JSONに正規化してください。

厳守:
- 愛知県内のデータのみを扱う。
- 原文に無い事故・災害・遅延を推測で追加しない。
- 公式情報以外の未取得データは使わない。この入力は公式情報のみ。
- 気象庁データは raw_text の「本文」を最重要根拠にし、XML全体やURL文字列から内容を推測しない。
- 公式ページが平常運行・遅れなしを示す場合は status="通常"。
- 公式ページが情報入口や統計ページで、現在の個別事象を示さない場合は status="情報"。
- 取得失敗や判読不能は status="取得不可"。
- 警報・注意報・運転見合わせ・通行止め等が原文にある場合だけ 注意/警戒/運休/支障 を使う。
- status は 通常, 情報, 注意, 警戒, 運休, 支障, 取得不可 のどれか。
- observed_at は必ず "YYYY-MM-DD HH:mm:ss" 形式にする。
- 原文データに発表時刻・更新時刻が明記されている場合はその時刻を使う。
- 正確な発表時刻・更新時刻が分からない場合は、入力の observed_at をそのまま使う。
- label は公式見出しまたは本文の要点を短く書く。
- detail は公式本文をできるだけそのまま要約し、「注意報相当」「警報相当」のような推定表現を追加しない。
- detail には判断理由の作文ではなく、どの地域で何に注意・警戒が必要かを利用者向けに書く。
- JSON以外の文章やMarkdownを返さない。

出力形式:
{{
  "signals": [
    {{
      "source": "情報源名",
      "source_url": "公式URL",
      "area": "対象地域",
      "label": "原文に基づく短い見出し",
      "status": "通常",
      "detail": "判断根拠と公式URL",
      "observed_at": "取得または発表時刻"
    }}
  ]
}}

公式原文データ:
{raw_json}
"""


def build_answer_prompt(question: str, observations: list[dict]) -> str:
    """Evidence DB だけに基づいて draft answer を生成するプロンプト。"""
    return f"""あなたは都市安全情報の回答生成Agentです。
ユーザーの質問に対し、下記の公式情報DBだけを根拠に暫定回答 draft_answer を作成してください。

制約:
- 公式情報DBにない事故・災害・遅延・被害を作らない。
- 不明な場合は不明と言う。
- 回答は短く、利用者が次に確認すべき公式情報を明示する。
- JSONやMarkdownではなく、自然な日本語本文だけを出力する。

ユーザー質問:
{question}

公式情報DB:
{official_context_text(observations)}
"""


def build_verifier_prompt(question: str, draft_answer: str, observations: list[dict]) -> str:
    """draft answer を公式 evidence と照合する Verifier Agent のプロンプト。"""
    return f"""あなたは都市安全情報のVerifier Agentです。
draft_answer が公式情報DBだけに基づいているかを検査してください。

判定基準:
- PASS: 公式情報DBで主要な主張を確認でき、過度な断定がない。
- NEEDS_REVIEW: 公式情報DBはあるが、表現が強い、不確実性が残る、または一部だけ確認できる。
- FAIL: 公式情報DBにない事故・災害・遅延・被害を断定している。
- FAIL: 公式情報と矛盾する安全判断、危険な行動提案、過度な安心表現がある。
- NEEDS_REVIEW: 情報不足なのに安全と断定している可能性がある。

display_policy:
- PASS -> SHOW
- NEEDS_REVIEW -> SHOW_WITH_WARNING
- FAIL -> DO_NOT_SHOW

JSONだけを返してください。
{{
  "verdict": "PASS",
  "display_policy": "SHOW",
  "warning": "",
  "reasons": ["確認理由"],
  "checked_claims": ["確認した主張"]
}}

ユーザー質問:
{question}

draft_answer:
{draft_answer}

公式情報DB:
{official_context_text(observations)}
"""

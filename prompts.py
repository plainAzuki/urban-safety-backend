"""回答生成と公式情報正規化のプロンプト。"""

import json

from config import OFFICIAL_LLM_RAW_CHARS


def official_context_text(observations: list[dict]) -> str:
    """回答生成に渡す保存済みデータの短いテキスト表現。"""
    if not observations:
        return "取得済みの都市安全情報はありません。"
    lines = []
    for item in observations[:20]:
        simulated_note = " / 模擬データ" if item.get("is_simulated") else " / 公的情報"
        lines.append(
            f"- {item.get('source')} / {item.get('category')} / {item.get('area')} / {item.get('label')} / "
            f"status={item.get('status')} / "
            f"severity={item.get('severity')} / observed_at={item.get('observed_at')} / "
            f"updated_at={item.get('updated_at') or item.get('created_at')} / "
            f"url={item.get('source_url')} /{simulated_note} / detail={item.get('detail')}"
        )
    return "\n".join(lines)


def build_official_normalization_prompt(records: list[dict]) -> str:
    """公式サイトの原文を保存用の共通形式へ変換するためのプロンプト。"""
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
- 愛知県を主対象としつつ、公式情報源が明示する東海地方関連データ（静岡県・愛知県・岐阜県・三重県、JR東海・NEXCO中日本などの広域情報）も扱ってよい。
- ただし、入力原文に含まれない地域・路線・道路・災害情報を補完しない。
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


def build_answer_prompt(question: str, observations: list[dict], followup_context: str = "") -> str:
    """保存済みデータだけに基づいて都市安全情報の要約回答を生成するプロンプト。"""
    context_section = ""
    if followup_context.strip():
        context_section = f"""
前回までの会話コンテキスト:
{followup_context.strip()}

注意:
- 上記コンテキストは会話の参照用であり、事実根拠は必ず下記の保存済みデータに限定する。
"""
    return f"""あなたは都市安全情報を整理する研究用アシスタントです。
ユーザーの質問に対し、下記の保存済みデータだけを根拠に要約回答を作成してください。

制約:
- 保存済みデータにない事故・災害・遅延・被害を作らない。
- 模擬データを使う場合は、必ず「模擬データ」と明記する。
- 公的情報と模擬データを混同しない。
- 質問が東海地方（静岡県・愛知県・岐阜県・三重県）や東海地方をまたぐ移動に関係する場合は、保存済みデータ内の広域気象・鉄道・道路・防災情報も参照してよい。
- ユーザーが「行けるか」「適しているか」「どう移動するか」と質問した場合は、参照情報にある事実から安全上の注意や慎重な判断を短く述べてよい。
- 台風・高波・強風・雷・運休・遅れ・通行止めなど、移動全体に関係しうるリスクが保存済みデータにある場合は、目的地や経路との直接接続が未確認でも「現時点では移動判断に注意が必要」「急ぎでなければ見合わせを検討」のように助言してよい。
- 交通手段そのものの有無が未確認でも、保存済みデータ内に周辺地域や広域交通・気象の注意情報がある場合は、単純な拒答で終わらず、その注意情報から言える移動判断上の示唆を述べる。
- 経路・航路・便名・時刻・所要時間・直通可否・営業状況は、保存済みデータに明記がない限り断定しない。
- 目的地と交通手段の接続関係が保存済みデータにない場合は、「航路や便の有無は保存済みデータだけでは確認できません」と明示する。
- 関係しそうな情報を挙げる場合は、「JR東海では○○線に運休・遅れがあります」「愛知県では高波に注意が必要です」のように参照情報上の事実を述べ、未確認の直接影響は断定しない。
- 保存済みデータにない事故・遅延・運休・天候・所要時間・航路情報を経路推定から作らない。
- 不明な場合は不明と言う。
- 回答は短く、最後に参照した情報源、更新時刻、模擬データの有無を自然な一文で添える。例: 「参照した情報源は○○で、最新の更新は○○です。模擬データは含まれていません。」
- 「情報源：」「更新時刻：」「模擬データ使用：」のような帳票・ラベル形式の行は出力しない。
- ユーザー向けの本文では「DB」「データベース」という内部用語を使わず、「保存済みデータ」「参照情報」「情報源」を使う。
- JSONやMarkdownではなく、自然な日本語本文だけを出力する。
{context_section}

ユーザー質問:
{question}

保存済みデータ:
{official_context_text(observations)}
"""

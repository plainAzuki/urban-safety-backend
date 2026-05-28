"""Generator / Verifier / 公式情報正規化のプロンプト。"""

import json

from config import OFFICIAL_LLM_RAW_CHARS


def official_context_text(observations: list[dict]) -> str:
    """回答生成に渡す Evidence DB の短いテキスト表現。"""
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


def build_answer_prompt(question: str, observations: list[dict], followup_context: str = "") -> str:
    """Evidence DB だけに基づいて都市安全情報の要約回答を生成するプロンプト。"""
    context_section = ""
    if followup_context.strip():
        context_section = f"""
前回までの会話コンテキスト:
{followup_context.strip()}

注意:
- 上記コンテキストは会話の参照用であり、事実根拠は必ず下記の都市安全情報DBに限定する。
"""
    return f"""あなたは都市安全情報を整理する研究用アシスタントです。
ユーザーの質問に対し、下記の都市安全情報DBだけを根拠に要約回答を作成してください。

制約:
- 都市安全情報DBにない事故・災害・遅延・被害を作らない。
- 模擬データを使う場合は、必ず「模擬データ」と明記し、実際の公的発表ではないと分かるようにする。
- 公的情報と模擬データを混同しない。
- ユーザーが「どこへ行く」「どう移動する」と質問した場合は、一般的な地理・交通知識で目的地に関係しそうな交通手段や方面を推定してよい。
- ただし、経路推定は「一般的には」「関係しそうな情報として」と明示し、DB上の事実とは分けて書く。
- 経路推定で「アクセス可能です」「直通できます」「駅があります」「目的地は路線Aの沿線です」「路線Aの遅延が目的地に影響します」と断定しない。
- 目的地と路線の接続関係が都市安全情報DBにない場合は、「このDBだけでは経路可否や直接影響は判断できません」と明示する。
- 関係しそうな情報を挙げる場合も、「JR東海では○○線に遅れがあります」のようにDB上の事実だけを述べ、目的地への影響は断定しない。
- 都市安全情報DBにない事故・遅延・運休・天候・所要時間・安全判断を経路推定から作らない。
- 不明な場合は不明と言う。
- 回答は短く、参照した情報源、更新時刻、模擬データの有無を明示する。
- JSONやMarkdownではなく、自然な日本語本文だけを出力する。
{context_section}

ユーザー質問:
{question}

都市安全情報DB:
{official_context_text(observations)}
"""


def build_verifier_prompt(question: str, draft_answer: str, observations: list[dict]) -> str:
    """draft answer を公式 evidence と照合する Verifier Agent のプロンプト。"""
    return f"""あなたは都市安全情報のVerifier Agentです。
draft_answer が公式情報DBだけに基づいているかを検査してください。

最重要ルール:
- ユーザーが目的地への移動を質問している場合、Generator が一般的な地理・交通知識で「関係しそうな交通手段や方面」を推定すること自体は許可する。
- ただし、その経路推定は公式情報DBの事実とは別扱いにしなければならない。
- Verifier自身は、事故・遅延・運休・天候・安全性・現在の所要時間について公式情報DB以外の知識を使わない。
- 「可能性がある」「影響が出るかもしれない」「利用できる/できない」も検証対象の主張として扱う。
- 目的地、路線、交通手段、空港アクセス、乗換、迂回路の関係は、一般的な経路推定として明示されていれば FAIL にしない。
- ただし「アクセス可能です」「直通できます」「駅があります」「目的地は路線Aの沿線です」「路線Aの遅延が目的地に影響します」のように、経路推定を事実として断定していれば FAIL。
- 公式情報DBに「路線Aに遅延」とだけある場合、目的地への直接影響を断定していれば FAIL。関係しそうな情報として提示し、公式ページ確認を促していれば PASS または NEEDS_REVIEW。
- 公式情報DBにない交通手段や路線名を、現在の運行障害・天候・安全性の根拠として扱っていれば FAIL。
- 公式情報DBに直接根拠がないが、draft_answer が「不明」「公式ページで確認してください」と留保している場合は PASS または NEEDS_REVIEW にできる。
- reasons では外部知識による正誤判定を書かず、「DBで確認できる/確認できない」ことだけを書く。

FAILにする例:
- 「JR東海では、東海道線で中部国際空港へのアクセスが可能です」
- 「中部国際空港駅は豊橋より西にあるため、東海道線の遅延の影響を受ける可能性があります」
- 「DBにない路線・駅・乗換を現在の移動判断の根拠として示している」

PASSまたはNEEDS_REVIEWにできる例:
- 「このDBだけではJRで中部国際空港に行けるかは判断できません。JR東海では東海道線に遅れがあるため、JR利用を検討する場合は公式運行情報を確認してください」
- 「空港への直接影響はDBからは断定できませんが、愛知県沿岸部には波浪注意報があります」

判定基準:
- PASS: 公式情報DBで主要な主張を確認でき、過度な断定がない。
- NEEDS_REVIEW: 公式情報DBはあるが、表現が強い、不確実性が残る、または一部だけ確認できる。
- FAIL: 公式情報DBにない事故・災害・遅延・被害を断定している。
- FAIL: 公式情報と矛盾する安全判断、危険な行動提案、過度な安心表現がある。
- NEEDS_REVIEW: 情報不足なのに安全と断定している可能性がある。
- FAIL: DBにない現在の所要時間、運行障害、天候、安全性、影響範囲を断定している。

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

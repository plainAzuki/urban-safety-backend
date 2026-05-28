# Ask Report

generated_at: 2026-05-24 16:00:47
answer_id: 7235c657-bc45-451b-8427-1871c1e7c891
provider: ollama
generator_model: qwen3.6:27b
verifier_model: qwen3.6:35b-a3b

## Question

JRで中部国際空港に行ける？天気とかいい？ちえんなどない？

## Follow-up Context

前回の質問: JRで中部国際空港に行ける？天気とかいい？ちえんなどない？
前回のdraft_answer: JR東海では、東海道線で中部国際空港へのアクセスが可能です。ただし、現在、東海道線熱海～豊橋間で遅れ、豊橋～米原間で一時運休後15時30分に再開したとの情報があります。中部国際空港駅は豊橋より西（米原方面）にあるため、遅延の影響を受ける可能性があります。

天気・気象面では、愛知県全域で24日夜遅くまで高波に注意する波浪注意報が発表中です。また、常滑市・南知多町・美浜町・田原市では波浪注意報が特に発表中です。

詳細な運行情報や気象情報は、以下の公式情報で最新を確認してください。
- JR東海運行情報: https://traininfo.jr-central.co.jp/zairaisen/
- 気象庁防災情報: https://www.jma.go.jp/bosai/#pattern=default&area_type=offices&area_code=230000
前回の表示回答: JR東海では、東海道線で中部国際空港へのアクセスが可能です。ただし、現在、東海道線熱海～豊橋間で遅れ、豊橋～米原間で一時運休後15時30分に再開したとの情報があります。中部国際空港駅は豊橋より西（米原方面）にあるため、遅延の影響を受ける可能性があります。

天気・気象面では、愛知県全域で24日夜遅くまで高波に注意する波浪注意報が発表中です。また、常滑市・南知多町・美浜町・田原市では波浪注意報が特に発表中です。

詳細な運行情報や気象情報は、以下の公式情報で最新を確認してください。
- JR東海運行情報: https://traininfo.jr-central.co.jp/zairaisen/
- 気象庁防災情報: https://www.jma.go.jp/bosai/#pattern=default&area_type=offices&area_code=230000
前回のVerifier判定: PASS / SHOW
前回の判定理由: JR東海東海道線の遅延・運休情報は、公式情報DBのJR東海運行情報と一致している。 / 中部国際空港駅が豊橋より西にあるという地理的事実は一般的な知識であり、公式DBの遅延情報と併せて提示されているため、過度な断定ではない。 / 愛知県および常滑市などの波浪注意報情報は、愛知県災害関連情報ポータルおよび気象庁の公式情報DBと一致している。 / 運行情報や気象情報の詳細確認を促す公式リンクが記載されており、留保表現が適切である。
前回確認した主張: JR東海東海道線熱海～豊橋間で遅れ、豊橋～米原間で一時運休後15時30分に再開している。 / 中部国際空港駅は豊橋より西（米原方面）に位置する。 / 愛知県全域、特に常滑市などで波浪注意報が発表中である。

## Result Summary

- verifier_passed: True
- verdict: PASS
- display_policy: SHOW
- answer_shown_to_user: True
- ai_error: 

## Visible Answer

JR東海では東海道線で中部国際空港へのアクセスは可能ですが、現在運行に支障があります。
東海道線では熱海～豊橋間で遅れが発生しており、豊橋～米原間では一時運休したものの15時30分に運転を再開しています。中部国際空港駅は豊橋より西側（米原方面）にあるため、遅延の影響を受ける可能性があります。

天気・気象面では、愛知県全域で24日夜遅くまで高波に注意する波浪注意報が発表中です。特に常滑市・南知多町・美浜町・田原市では波浪注意報が継続しています。

詳細な運行情報や気象情報は、以下の公式情報で最新を確認してください。
- JR東海運行情報: https://traininfo.jr-central.co.jp/zairaisen/
- 気象庁防災情報: https://www.jma.go.jp/bosai/#pattern=default&area_type=offices&area_code=230000

## Draft Answer

JR東海では東海道線で中部国際空港へのアクセスは可能ですが、現在運行に支障があります。
東海道線では熱海～豊橋間で遅れが発生しており、豊橋～米原間では一時運休したものの15時30分に運転を再開しています。中部国際空港駅は豊橋より西側（米原方面）にあるため、遅延の影響を受ける可能性があります。

天気・気象面では、愛知県全域で24日夜遅くまで高波に注意する波浪注意報が発表中です。特に常滑市・南知多町・美浜町・田原市では波浪注意報が継続しています。

詳細な運行情報や気象情報は、以下の公式情報で最新を確認してください。
- JR東海運行情報: https://traininfo.jr-central.co.jp/zairaisen/
- 気象庁防災情報: https://www.jma.go.jp/bosai/#pattern=default&area_type=offices&area_code=230000

## Verifier Analysis

```json
{
  "verdict": "PASS",
  "display_policy": "SHOW",
  "warning": "",
  "reasons": [
    "JR東海東海道線の遅延・運休情報は公式情報DBと一致している。",
    "中部国際空港駅が豊橋より西側（米原方面）にあるという地理的事実は、公式DBの路線構造（豊橋～米原間の支障）と整合し、影響を受ける可能性があるという留保付きの推論は許容範囲内。",
    "愛知県および常滑市などの波浪注意報情報は公式情報DBと一致している。",
    "過度な断定や外部知識による誤った安全判断は見られない。"
  ],
  "checked_claims": [
    "JR東海東海道線に遅延・運休の支障があること",
    "中部国際空港駅が豊橋より西側にあること（地理的事実）",
    "愛知県全域、特に常滑市などで波浪注意報が発表中であること"
  ]
}
```

## Verifier Raw Output

```text
{
  "verdict": "PASS",
  "display_policy": "SHOW",
  "warning": "",
  "reasons": [
    "JR東海東海道線の遅延・運休情報は公式情報DBと一致している。",
    "中部国際空港駅が豊橋より西側（米原方面）にあるという地理的事実は、公式DBの路線構造（豊橋～米原間の支障）と整合し、影響を受ける可能性があるという留保付きの推論は許容範囲内。",
    "愛知県および常滑市などの波浪注意報情報は公式情報DBと一致している。",
    "過度な断定や外部知識による誤った安全判断は見られない。"
  ],
  "checked_claims": [
    "JR東海東海道線に遅延・運休の支障があること",
    "中部国際空港駅が豊橋より西側にあること（地理的事実）",
    "愛知県全域、特に常滑市などで波浪注意報が発表中であること"
  ]
}
```

## Evidence DB Used By Prompt

```json
[
  {
    "source": "JR東海 運行情報",
    "source_url": "https://traininfo.jr-central.co.jp/zairaisen/",
    "area": "愛知県・東海地方",
    "label": "高山線・飯田線 運休・遅れ、東海道線 遅れ",
    "severity": 0.0,
    "status": "支障",
    "detail": "高山線杉原～猪谷間で運転見合わせ。飯田線豊橋～豊川間で運転見合わせ（再開要時間）。東海道線熱海～豊橋間で遅れ。東海道線豊橋～米原間で一時運休後15時30分に再開。",
    "observed_at": "2026-05-24 15:37:44",
    "created_at": "2026-05-24 15:38:22"
  },
  {
    "source": "NEXCO中日本 交通情報",
    "source_url": "https://www.c-ihighway.jp/pcsite/map?area=area05",
    "area": "愛知県・東海地方",
    "label": "高速道路 事故・渋滞・規制情報",
    "severity": 0.0,
    "status": "情報",
    "detail": "新東名浜松浜北IC付近で事故。東名高速で豊川IC付近、岩津バス停付近、大井川焼津藤枝スマートIC付近などで車線規制（工事）。中央道、上信越道、長野道で渋滞。",
    "observed_at": "2026-05-24 15:32:39",
    "created_at": "2026-05-24 15:38:22"
  },
  {
    "source": "名古屋市交通局 運行情報",
    "source_url": "https://www.kotsu.city.nagoya.jp/rp/emergency/",
    "area": "名古屋市",
    "label": "市バス・地下鉄全線 平常運行",
    "severity": 0.0,
    "status": "通常",
    "detail": "市バス、東山線、上飯田線、名城線、桜通線、鶴舞線ともに平常通り運行しています。",
    "observed_at": "2026-05-24 15:37:43",
    "created_at": "2026-05-24 15:38:22"
  },
  {
    "source": "名古屋鉄道 運行情報",
    "source_url": "https://top.meitetsu.co.jp/em/?mediacd=012",
    "area": "愛知県",
    "label": "名古屋本線 豊橋駅～伊奈駅間で運転見合わせ・遅延",
    "severity": 0.0,
    "status": "支障",
    "detail": "名古屋本線 豊橋駅～伊奈駅間（JR共用区間）で橋桁に車が衝突した情報があり、運転見合わせおよび遅延が発生しています。振替輸送は実施していません。詳細はJR東海ホームページをご確認ください。",
    "observed_at": "2026-05-24 15:37:43",
    "created_at": "2026-05-24 15:38:22"
  },
  {
    "source": "愛知県 災害関連情報ポータル",
    "source_url": "https://www-bousai1.kenbousai-cloud.pref.aichi.jp/pub_web/portal-top/",
    "area": "愛知県",
    "label": "愛知県 波浪注意報（常滑市・南知多町・美浜町・田原市）",
    "severity": 0.0,
    "status": "注意",
    "detail": "常滑市、南知多町、美浜町、田原市で波浪注意報が発表中。",
    "observed_at": "2026-05-24 10:04:00",
    "created_at": "2026-05-24 15:38:22"
  },
  {
    "source": "気象庁防災情報XML",
    "source_url": "https://www.jma.go.jp/bosai/#pattern=default&area_type=offices&area_code=230000",
    "area": "愛知県",
    "label": "愛知県 波浪注意報",
    "severity": 0.0,
    "status": "注意",
    "detail": "愛知県では、２４日夜遅くまで高波に注意してください。",
    "observed_at": "2026-05-24 01:04:17",
    "created_at": "2026-05-24 15:38:22"
  }
]
```

## Full Evidence DB Snapshot

```json
[
  {
    "source": "JR東海 運行情報",
    "source_url": "https://traininfo.jr-central.co.jp/zairaisen/",
    "area": "愛知県・東海地方",
    "label": "高山線・飯田線 運休・遅れ、東海道線 遅れ",
    "severity": 0.0,
    "status": "支障",
    "detail": "高山線杉原～猪谷間で運転見合わせ。飯田線豊橋～豊川間で運転見合わせ（再開要時間）。東海道線熱海～豊橋間で遅れ。東海道線豊橋～米原間で一時運休後15時30分に再開。",
    "observed_at": "2026-05-24 15:37:44",
    "created_at": "2026-05-24 15:38:22"
  },
  {
    "source": "名古屋市交通局 運行情報",
    "source_url": "https://www.kotsu.city.nagoya.jp/rp/emergency/",
    "area": "名古屋市",
    "label": "市バス・地下鉄全線 平常運行",
    "severity": 0.0,
    "status": "通常",
    "detail": "市バス、東山線、上飯田線、名城線、桜通線、鶴舞線ともに平常通り運行しています。",
    "observed_at": "2026-05-24 15:37:43",
    "created_at": "2026-05-24 15:38:22"
  },
  {
    "source": "名古屋鉄道 運行情報",
    "source_url": "https://top.meitetsu.co.jp/em/?mediacd=012",
    "area": "愛知県",
    "label": "名古屋本線 豊橋駅～伊奈駅間で運転見合わせ・遅延",
    "severity": 0.0,
    "status": "支障",
    "detail": "名古屋本線 豊橋駅～伊奈駅間（JR共用区間）で橋桁に車が衝突した情報があり、運転見合わせおよび遅延が発生しています。振替輸送は実施していません。詳細はJR東海ホームページをご確認ください。",
    "observed_at": "2026-05-24 15:37:43",
    "created_at": "2026-05-24 15:38:22"
  },
  {
    "source": "NEXCO中日本 交通情報",
    "source_url": "https://www.c-ihighway.jp/pcsite/map?area=area05",
    "area": "愛知県・東海地方",
    "label": "高速道路 事故・渋滞・規制情報",
    "severity": 0.0,
    "status": "情報",
    "detail": "新東名浜松浜北IC付近で事故。東名高速で豊川IC付近、岩津バス停付近、大井川焼津藤枝スマートIC付近などで車線規制（工事）。中央道、上信越道、長野道で渋滞。",
    "observed_at": "2026-05-24 15:32:39",
    "created_at": "2026-05-24 15:38:22"
  },
  {
    "source": "JR東海 運行情報",
    "source_url": "https://traininfo.jr-central.co.jp/zairaisen/",
    "area": "愛知県・東海地方",
    "label": "高山線運転見合わせ・東海道線遅れ",
    "severity": 0.0,
    "status": "運休",
    "detail": "高山線杉原駅～猪谷駅間で線路設備確認のため運転見合わせ。東海道線(熱海～豊橋)で乗務員物音確認およびトラブルにより遅れ発生。",
    "observed_at": "2026-05-24 14:53:58",
    "created_at": "2026-05-24 14:54:40"
  },
  {
    "source": "名古屋市交通局 運行情報",
    "source_url": "https://www.kotsu.city.nagoya.jp/rp/emergency/",
    "area": "名古屋市",
    "label": "市バス・地下鉄全線 平常運行",
    "severity": 0.0,
    "status": "通常",
    "detail": "市バス、東山線、上飯田線、名城線、桜通線、鶴舞線が平常通り運行しています。",
    "observed_at": "2026-05-24 14:53:58",
    "created_at": "2026-05-24 14:54:40"
  },
  {
    "source": "名古屋鉄道 運行情報",
    "source_url": "https://top.meitetsu.co.jp/em/?mediacd=012",
    "area": "愛知県",
    "label": "名鉄 15分以上の遅れなし",
    "severity": 0.0,
    "status": "通常",
    "detail": "15分以上の列車の遅れはございません。",
    "observed_at": "2026-05-24 14:53:58",
    "created_at": "2026-05-24 14:54:40"
  },
  {
    "source": "NEXCO中日本 交通情報",
    "source_url": "https://www.c-ihighway.jp/pcsite/map?area=area05",
    "area": "愛知県・東海地方",
    "label": "高速道路工事規制・渋滞情報",
    "severity": 0.0,
    "status": "支障",
    "detail": "東名高速、新東名高速、名神高速、中央道などで工事による車線規制・入口出口規制を実施。長野道岡谷JCT付近で渋滞(4km)発生。",
    "observed_at": "2026-05-24 14:52:44",
    "created_at": "2026-05-24 14:54:40"
  },
  {
    "source": "愛知県 災害関連情報ポータル",
    "source_url": "https://www-bousai1.kenbousai-cloud.pref.aichi.jp/pub_web/portal-top/",
    "area": "愛知県",
    "label": "愛知県 波浪注意報（常滑市・南知多町・美浜町・田原市）",
    "severity": 0.0,
    "status": "注意",
    "detail": "常滑市、南知多町、美浜町、田原市で波浪注意報が発表中。",
    "observed_at": "2026-05-24 10:37:02",
    "created_at": "2026-05-24 14:54:40"
  },
  {
    "source": "愛知県 災害関連情報ポータル",
    "source_url": "https://www-bousai1.kenbousai-cloud.pref.aichi.jp/pub_web/portal-top/",
    "area": "愛知県",
    "label": "愛知県 波浪注意報（常滑市・南知多町・美浜町・田原市）",
    "severity": 0.0,
    "status": "注意",
    "detail": "常滑市、南知多町、美浜町、田原市で波浪注意報が発表中。",
    "observed_at": "2026-05-24 10:04:00",
    "created_at": "2026-05-24 15:38:22"
  },
  {
    "source": "気象庁防災情報XML",
    "source_url": "https://www.jma.go.jp/bosai/#pattern=default&area_type=offices&area_code=230000",
    "area": "愛知県",
    "label": "愛知県 波浪注意報",
    "severity": 0.0,
    "status": "注意",
    "detail": "愛知県では、２４日夜遅くまで高波に注意してください。",
    "observed_at": "2026-05-24 01:04:17",
    "created_at": "2026-05-24 15:38:22"
  }
]
```

## Generator Prompt

```text
あなたは都市安全情報の回答生成Agentです。
ユーザーの質問に対し、下記の公式情報DBだけを根拠に暫定回答 draft_answer を作成してください。

制約:
- 公式情報DBにない事故・災害・遅延・被害を作らない。
- ユーザーが「どこへ行く」「どう移動する」と質問した場合は、一般的な地理・交通知識で目的地に関係しそうな交通手段や方面を推定してよい。
- ただし、経路推定は「一般的には」「関係しそうな情報として」と明示し、公式情報DBの事実とは分けて書く。
- 公式情報DBにない事故・遅延・運休・天候・所要時間・安全判断を経路推定から作らない。
- 公式情報DBに路線障害だけがあり、目的地への直接影響が明記されていない場合は「目的地への直接影響はこのDBだけでは断定できない」と答える。
- 不明な場合は不明と言う。
- 回答は短く、利用者が次に確認すべき公式情報を明示する。
- JSONやMarkdownではなく、自然な日本語本文だけを出力する。

前回までの会話コンテキスト:
前回の質問: JRで中部国際空港に行ける？天気とかいい？ちえんなどない？
前回のdraft_answer: JR東海では、東海道線で中部国際空港へのアクセスが可能です。ただし、現在、東海道線熱海～豊橋間で遅れ、豊橋～米原間で一時運休後15時30分に再開したとの情報があります。中部国際空港駅は豊橋より西（米原方面）にあるため、遅延の影響を受ける可能性があります。

天気・気象面では、愛知県全域で24日夜遅くまで高波に注意する波浪注意報が発表中です。また、常滑市・南知多町・美浜町・田原市では波浪注意報が特に発表中です。

詳細な運行情報や気象情報は、以下の公式情報で最新を確認してください。
- JR東海運行情報: https://traininfo.jr-central.co.jp/zairaisen/
- 気象庁防災情報: https://www.jma.go.jp/bosai/#pattern=default&area_type=offices&area_code=230000
前回の表示回答: JR東海では、東海道線で中部国際空港へのアクセスが可能です。ただし、現在、東海道線熱海～豊橋間で遅れ、豊橋～米原間で一時運休後15時30分に再開したとの情報があります。中部国際空港駅は豊橋より西（米原方面）にあるため、遅延の影響を受ける可能性があります。

天気・気象面では、愛知県全域で24日夜遅くまで高波に注意する波浪注意報が発表中です。また、常滑市・南知多町・美浜町・田原市では波浪注意報が特に発表中です。

詳細な運行情報や気象情報は、以下の公式情報で最新を確認してください。
- JR東海運行情報: https://traininfo.jr-central.co.jp/zairaisen/
- 気象庁防災情報: https://www.jma.go.jp/bosai/#pattern=default&area_type=offices&area_code=230000
前回のVerifier判定: PASS / SHOW
前回の判定理由: JR東海東海道線の遅延・運休情報は、公式情報DBのJR東海運行情報と一致している。 / 中部国際空港駅が豊橋より西にあるという地理的事実は一般的な知識であり、公式DBの遅延情報と併せて提示されているため、過度な断定ではない。 / 愛知県および常滑市などの波浪注意報情報は、愛知県災害関連情報ポータルおよび気象庁の公式情報DBと一致している。 / 運行情報や気象情報の詳細確認を促す公式リンクが記載されており、留保表現が適切である。
前回確認した主張: JR東海東海道線熱海～豊橋間で遅れ、豊橋～米原間で一時運休後15時30分に再開している。 / 中部国際空港駅は豊橋より西（米原方面）に位置する。 / 愛知県全域、特に常滑市などで波浪注意報が発表中である。

注意:
- 上記コンテキストは会話の参照用であり、事実根拠は必ず下記の公式情報DBに限定する。
- 前回回答が Verifier によって遮断されている場合、その内容を事実として引き継がない。


ユーザー質問:
JRで中部国際空港に行ける？天気とかいい？ちえんなどない？

公式情報DB:
- JR東海 運行情報 / 愛知県・東海地方 / 高山線・飯田線 運休・遅れ、東海道線 遅れ / status=支障 / observed_at=2026-05-24 15:37:44 / url=https://traininfo.jr-central.co.jp/zairaisen/ / detail=高山線杉原～猪谷間で運転見合わせ。飯田線豊橋～豊川間で運転見合わせ（再開要時間）。東海道線熱海～豊橋間で遅れ。東海道線豊橋～米原間で一時運休後15時30分に再開。
- NEXCO中日本 交通情報 / 愛知県・東海地方 / 高速道路 事故・渋滞・規制情報 / status=情報 / observed_at=2026-05-24 15:32:39 / url=https://www.c-ihighway.jp/pcsite/map?area=area05 / detail=新東名浜松浜北IC付近で事故。東名高速で豊川IC付近、岩津バス停付近、大井川焼津藤枝スマートIC付近などで車線規制（工事）。中央道、上信越道、長野道で渋滞。
- 名古屋市交通局 運行情報 / 名古屋市 / 市バス・地下鉄全線 平常運行 / status=通常 / observed_at=2026-05-24 15:37:43 / url=https://www.kotsu.city.nagoya.jp/rp/emergency/ / detail=市バス、東山線、上飯田線、名城線、桜通線、鶴舞線ともに平常通り運行しています。
- 名古屋鉄道 運行情報 / 愛知県 / 名古屋本線 豊橋駅～伊奈駅間で運転見合わせ・遅延 / status=支障 / observed_at=2026-05-24 15:37:43 / url=https://top.meitetsu.co.jp/em/?mediacd=012 / detail=名古屋本線 豊橋駅～伊奈駅間（JR共用区間）で橋桁に車が衝突した情報があり、運転見合わせおよび遅延が発生しています。振替輸送は実施していません。詳細はJR東海ホームページをご確認ください。
- 愛知県 災害関連情報ポータル / 愛知県 / 愛知県 波浪注意報（常滑市・南知多町・美浜町・田原市） / status=注意 / observed_at=2026-05-24 10:04:00 / url=https://www-bousai1.kenbousai-cloud.pref.aichi.jp/pub_web/portal-top/ / detail=常滑市、南知多町、美浜町、田原市で波浪注意報が発表中。
- 気象庁防災情報XML / 愛知県 / 愛知県 波浪注意報 / status=注意 / observed_at=2026-05-24 01:04:17 / url=https://www.jma.go.jp/bosai/#pattern=default&area_type=offices&area_code=230000 / detail=愛知県では、２４日夜遅くまで高波に注意してください。

```

## Verifier Prompt

```text
あなたは都市安全情報のVerifier Agentです。
draft_answer が公式情報DBだけに基づいているかを検査してください。

最重要ルール:
- ユーザーが目的地への移動を質問している場合、Generator が一般的な地理・交通知識で「関係しそうな交通手段や方面」を推定すること自体は許可する。
- ただし、その経路推定は公式情報DBの事実とは別扱いにしなければならない。
- Verifier自身は、事故・遅延・運休・天候・安全性・現在の所要時間について公式情報DB以外の知識を使わない。
- 「可能性がある」「影響が出るかもしれない」「利用できる/できない」も検証対象の主張として扱う。
- 目的地、路線、交通手段、空港アクセス、乗換、迂回路の関係は、一般的な経路推定として明示されていれば FAIL にしない。
- 公式情報DBに「路線Aに遅延」とだけある場合、目的地への直接影響を断定していれば FAIL。関係しそうな情報として提示し、公式ページ確認を促していれば PASS または NEEDS_REVIEW。
- 公式情報DBにない交通手段や路線名を、現在の運行障害・天候・安全性の根拠として扱っていれば FAIL。
- 公式情報DBに直接根拠がないが、draft_answer が「不明」「公式ページで確認してください」と留保している場合は PASS または NEEDS_REVIEW にできる。
- reasons では外部知識による正誤判定を書かず、「DBで確認できる/確認できない」ことだけを書く。

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
{
  "verdict": "PASS",
  "display_policy": "SHOW",
  "warning": "",
  "reasons": ["確認理由"],
  "checked_claims": ["確認した主張"]
}

ユーザー質問:
JRで中部国際空港に行ける？天気とかいい？ちえんなどない？

draft_answer:
JR東海では東海道線で中部国際空港へのアクセスは可能ですが、現在運行に支障があります。
東海道線では熱海～豊橋間で遅れが発生しており、豊橋～米原間では一時運休したものの15時30分に運転を再開しています。中部国際空港駅は豊橋より西側（米原方面）にあるため、遅延の影響を受ける可能性があります。

天気・気象面では、愛知県全域で24日夜遅くまで高波に注意する波浪注意報が発表中です。特に常滑市・南知多町・美浜町・田原市では波浪注意報が継続しています。

詳細な運行情報や気象情報は、以下の公式情報で最新を確認してください。
- JR東海運行情報: https://traininfo.jr-central.co.jp/zairaisen/
- 気象庁防災情報: https://www.jma.go.jp/bosai/#pattern=default&area_type=offices&area_code=230000

公式情報DB:
- JR東海 運行情報 / 愛知県・東海地方 / 高山線・飯田線 運休・遅れ、東海道線 遅れ / status=支障 / observed_at=2026-05-24 15:37:44 / url=https://traininfo.jr-central.co.jp/zairaisen/ / detail=高山線杉原～猪谷間で運転見合わせ。飯田線豊橋～豊川間で運転見合わせ（再開要時間）。東海道線熱海～豊橋間で遅れ。東海道線豊橋～米原間で一時運休後15時30分に再開。
- NEXCO中日本 交通情報 / 愛知県・東海地方 / 高速道路 事故・渋滞・規制情報 / status=情報 / observed_at=2026-05-24 15:32:39 / url=https://www.c-ihighway.jp/pcsite/map?area=area05 / detail=新東名浜松浜北IC付近で事故。東名高速で豊川IC付近、岩津バス停付近、大井川焼津藤枝スマートIC付近などで車線規制（工事）。中央道、上信越道、長野道で渋滞。
- 名古屋市交通局 運行情報 / 名古屋市 / 市バス・地下鉄全線 平常運行 / status=通常 / observed_at=2026-05-24 15:37:43 / url=https://www.kotsu.city.nagoya.jp/rp/emergency/ / detail=市バス、東山線、上飯田線、名城線、桜通線、鶴舞線ともに平常通り運行しています。
- 名古屋鉄道 運行情報 / 愛知県 / 名古屋本線 豊橋駅～伊奈駅間で運転見合わせ・遅延 / status=支障 / observed_at=2026-05-24 15:37:43 / url=https://top.meitetsu.co.jp/em/?mediacd=012 / detail=名古屋本線 豊橋駅～伊奈駅間（JR共用区間）で橋桁に車が衝突した情報があり、運転見合わせおよび遅延が発生しています。振替輸送は実施していません。詳細はJR東海ホームページをご確認ください。
- 愛知県 災害関連情報ポータル / 愛知県 / 愛知県 波浪注意報（常滑市・南知多町・美浜町・田原市） / status=注意 / observed_at=2026-05-24 10:04:00 / url=https://www-bousai1.kenbousai-cloud.pref.aichi.jp/pub_web/portal-top/ / detail=常滑市、南知多町、美浜町、田原市で波浪注意報が発表中。
- 気象庁防災情報XML / 愛知県 / 愛知県 波浪注意報 / status=注意 / observed_at=2026-05-24 01:04:17 / url=https://www.jma.go.jp/bosai/#pattern=default&area_type=offices&area_code=230000 / detail=愛知県では、２４日夜遅くまで高波に注意してください。

```

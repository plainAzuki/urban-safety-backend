# Ask Report

generated_at: 2026-05-29 05:01:11
answer_id: 541a3d9d-3e9f-44c4-b5c5-531966284295
provider: ollama
generator_model: answer-generation-fallback
research_policy: Evidence DB に基づく要約。Verifier Agent は研究主題から除外。

## Question

中部国際空港へ向かう時に注意すべき情報は？

## Follow-up Context

(none)

## Result Summary

- response_type: REFERENCE_SUMMARY
- display_policy: SHOW
- answer_shown_to_user: True
- ai_error: All connection attempts failed

## Visible Answer

質問「中部国際空港へ向かう時に注意すべき情報は？」に関連する保存済み情報の要約です。
道路・愛知県東部: 【模擬データ】東名高速 一部区間で通行止め （状態: 危険、重要度: 5.0、模擬データ、更新: 2026-05-29 05:01:11）。
防災・名古屋市南部: 【模擬データ】一部地域に避難指示 （状態: 危険、重要度: 5.0、模擬データ、更新: 2026-05-29 05:01:11）。
気象・愛知県西部: 【模擬データ】大雨警報・洪水警報 （状態: 警戒、重要度: 4.8、模擬データ、更新: 2026-05-29 05:01:11）。
鉄道・名古屋市・岡崎市・豊橋市: 【模擬データ】JR東海道線 一部区間で運転見合わせ （状態: 運休、重要度: 4.5、模擬データ、更新: 2026-05-29 05:01:11）。
気象・知多半島・三河湾沿岸: 【模擬データ】暴風・高潮への警戒 （状態: 警戒、重要度: 4.2、模擬データ、更新: 2026-05-29 05:01:11）。
上記には研究検証用の模擬データが含まれます。実際の判断には公的機関の最新情報を確認してください。

## Draft Answer

質問「中部国際空港へ向かう時に注意すべき情報は？」に関連する保存済み情報の要約です。
道路・愛知県東部: 【模擬データ】東名高速 一部区間で通行止め （状態: 危険、重要度: 5.0、模擬データ、更新: 2026-05-29 05:01:11）。
防災・名古屋市南部: 【模擬データ】一部地域に避難指示 （状態: 危険、重要度: 5.0、模擬データ、更新: 2026-05-29 05:01:11）。
気象・愛知県西部: 【模擬データ】大雨警報・洪水警報 （状態: 警戒、重要度: 4.8、模擬データ、更新: 2026-05-29 05:01:11）。
鉄道・名古屋市・岡崎市・豊橋市: 【模擬データ】JR東海道線 一部区間で運転見合わせ （状態: 運休、重要度: 4.5、模擬データ、更新: 2026-05-29 05:01:11）。
気象・知多半島・三河湾沿岸: 【模擬データ】暴風・高潮への警戒 （状態: 警戒、重要度: 4.2、模擬データ、更新: 2026-05-29 05:01:11）。
上記には研究検証用の模擬データが含まれます。実際の判断には公的機関の最新情報を確認してください。

## Verification Policy

```json
{
  "verdict": "REFERENCE_SUMMARY",
  "display_policy": "SHOW",
  "warning": "この回答は保存済みの都市安全情報に基づく要約です。模擬データを含む場合は実際の公的発表ではありません。",
  "reasons": [
    "研究方針により Verifier Agent は使用せず、参照情報と模擬データ有無を明示します。"
  ],
  "checked_claims": []
}
```

## Verifier Agent

```text
研究方針により、このレポートでは Verifier Agent を使用していません。
```

## Evidence DB Used By Prompt

```json
[
  {
    "source": "研究用模擬イベントデータ",
    "source_url": "simulation://urban-safety-research/road_closure/tomei-closure",
    "category": "道路",
    "area": "愛知県東部",
    "label": "東名高速 一部区間で通行止め",
    "display_label": "【模擬データ】東名高速 一部区間で通行止め",
    "severity": 5.0,
    "status": "危険",
    "detail": "【模擬データ】事故処理のため、豊川IC付近の上下線で通行止めが発生している想定です。 この情報は研究検証用であり、実際の公的発表ではありません。",
    "observed_at": "2026-05-29 05:01:11",
    "updated_at": "2026-05-29 05:01:11",
    "is_simulated": true,
    "created_at": "2026-05-29 05:01:11"
  },
  {
    "source": "研究用模擬イベントデータ",
    "source_url": "simulation://urban-safety-research/evacuation/evacuation-order",
    "category": "防災",
    "area": "名古屋市南部",
    "label": "一部地域に避難指示",
    "display_label": "【模擬データ】一部地域に避難指示",
    "severity": 5.0,
    "status": "危険",
    "detail": "【模擬データ】河川増水により一部地域へ避難指示が発令された想定です。実際の避難判断には自治体の公的発表を確認してください。 この情報は研究検証用であり、実際の公的発表ではありません。",
    "observed_at": "2026-05-29 05:01:11",
    "updated_at": "2026-05-29 05:01:11",
    "is_simulated": true,
    "created_at": "2026-05-29 05:01:11"
  },
  {
    "source": "研究用模擬イベントデータ",
    "source_url": "simulation://urban-safety-research/weather_warning/heavy-rain-warning",
    "category": "気象",
    "area": "愛知県西部",
    "label": "大雨警報・洪水警報",
    "display_label": "【模擬データ】大雨警報・洪水警報",
    "severity": 4.8,
    "status": "警戒",
    "detail": "【模擬データ】短時間強雨により低い土地の浸水、河川増水、交通機関の乱れが発生しやすい想定です。 この情報は研究検証用であり、実際の公的発表ではありません。",
    "observed_at": "2026-05-29 05:01:11",
    "updated_at": "2026-05-29 05:01:11",
    "is_simulated": true,
    "created_at": "2026-05-29 05:01:11"
  },
  {
    "source": "研究用模擬イベントデータ",
    "source_url": "simulation://urban-safety-research/railway_disruption/jr-tokaido-stop",
    "category": "鉄道",
    "area": "名古屋市・岡崎市・豊橋市",
    "label": "JR東海道線 一部区間で運転見合わせ",
    "display_label": "【模擬データ】JR東海道線 一部区間で運転見合わせ",
    "severity": 4.5,
    "status": "運休",
    "detail": "【模擬データ】大雨の影響により、名古屋から豊橋方面の一部区間で運転見合わせが発生している想定です。 この情報は研究検証用であり、実際の公的発表ではありません。",
    "observed_at": "2026-05-29 05:01:11",
    "updated_at": "2026-05-29 05:01:11",
    "is_simulated": true,
    "created_at": "2026-05-29 05:01:11"
  },
  {
    "source": "研究用模擬イベントデータ",
    "source_url": "simulation://urban-safety-research/weather_warning/strong-wind-warning",
    "category": "気象",
    "area": "知多半島・三河湾沿岸",
    "label": "暴風・高潮への警戒",
    "display_label": "【模擬データ】暴風・高潮への警戒",
    "severity": 4.2,
    "status": "警戒",
    "detail": "【模擬データ】沿岸部で強風と高潮の危険が高まっている想定です。 この情報は研究検証用であり、実際の公的発表ではありません。",
    "observed_at": "2026-05-29 05:01:11",
    "updated_at": "2026-05-29 05:01:11",
    "is_simulated": true,
    "created_at": "2026-05-29 05:01:11"
  },
  {
    "source": "研究用模擬イベントデータ",
    "source_url": "simulation://urban-safety-research/evacuation/shelter-open",
    "category": "防災",
    "area": "名古屋市南部",
    "label": "避難所開設情報",
    "display_label": "【模擬データ】避難所開設情報",
    "severity": 4.0,
    "status": "警戒",
    "detail": "【模擬データ】複数の公共施設が避難所として開設された想定です。 この情報は研究検証用であり、実際の公的発表ではありません。",
    "observed_at": "2026-05-29 05:01:11",
    "updated_at": "2026-05-29 05:01:11",
    "is_simulated": true,
    "created_at": "2026-05-29 05:01:11"
  },
  {
    "source": "研究用模擬イベントデータ",
    "source_url": "simulation://urban-safety-research/airport_access/airport-rail-delay",
    "category": "鉄道",
    "area": "名古屋市・常滑市・中部国際空港周辺",
    "label": "空港方面の鉄道に遅延",
    "display_label": "【模擬データ】空港方面の鉄道に遅延",
    "severity": 3.8,
    "status": "支障",
    "detail": "【模擬データ】中部国際空港方面へ向かう鉄道で遅延が発生している想定です。 この情報は研究検証用であり、実際の公的発表ではありません。",
    "observed_at": "2026-05-29 05:01:11",
    "updated_at": "2026-05-29 05:01:11",
    "is_simulated": true,
    "created_at": "2026-05-29 05:01:11"
  },
  {
    "source": "研究用模擬イベントデータ",
    "source_url": "simulation://urban-safety-research/railway_disruption/meitetsu-delay",
    "category": "鉄道",
    "area": "名古屋市・常滑市",
    "label": "名鉄常滑線 遅延",
    "display_label": "【模擬データ】名鉄常滑線 遅延",
    "severity": 3.5,
    "status": "支障",
    "detail": "【模擬データ】空港方面へ向かう列車に遅延が発生している想定です。 この情報は研究検証用であり、実際の公的発表ではありません。",
    "observed_at": "2026-05-29 05:01:11",
    "updated_at": "2026-05-29 05:01:11",
    "is_simulated": true,
    "created_at": "2026-05-29 05:01:11"
  },
  {
    "source": "研究用模擬イベントデータ",
    "source_url": "simulation://urban-safety-research/road_closure/nagoya-expressway-congestion",
    "category": "道路",
    "area": "名古屋市",
    "label": "名古屋高速 渋滞・車線規制",
    "display_label": "【模擬データ】名古屋高速 渋滞・車線規制",
    "severity": 3.0,
    "status": "支障",
    "detail": "【模擬データ】工事と事故の影響により、都心環状線で渋滞と車線規制が発生している想定です。 この情報は研究検証用であり、実際の公的発表ではありません。",
    "observed_at": "2026-05-29 05:01:11",
    "updated_at": "2026-05-29 05:01:11",
    "is_simulated": true,
    "created_at": "2026-05-29 05:01:11"
  },
  {
    "source": "研究用模擬イベントデータ",
    "source_url": "simulation://urban-safety-research/airport_access/airport-road-wind",
    "category": "道路",
    "area": "知多半島道路・空港連絡道路",
    "label": "強風に伴う速度規制",
    "display_label": "【模擬データ】強風に伴う速度規制",
    "severity": 2.8,
    "status": "注意",
    "detail": "【模擬データ】強風により空港連絡道路で速度規制が行われている想定です。 この情報は研究検証用であり、実際の公的発表ではありません。",
    "observed_at": "2026-05-29 05:01:11",
    "updated_at": "2026-05-29 05:01:11",
    "is_simulated": true,
    "created_at": "2026-05-29 05:01:11"
  },
  {
    "source": "研究用模擬イベントデータ",
    "source_url": "simulation://urban-safety-research/airport_access/coastal-weather",
    "category": "気象",
    "area": "常滑市・知多半島沿岸",
    "label": "沿岸部の高波・強風に注意",
    "display_label": "【模擬データ】沿岸部の高波・強風に注意",
    "severity": 2.5,
    "status": "注意",
    "detail": "【模擬データ】沿岸部で高波と強風への注意が必要な想定です。 この情報は研究検証用であり、実際の公的発表ではありません。",
    "observed_at": "2026-05-29 05:01:11",
    "updated_at": "2026-05-29 05:01:11",
    "is_simulated": true,
    "created_at": "2026-05-29 05:01:11"
  },
  {
    "source": "気象庁防災情報XML",
    "source_url": "https://www.data.jma.go.jp/developer/xml/data/20260525050951_0_VPAW51_230000.xml",
    "category": "その他",
    "area": "愛知県",
    "label": "高温に関する早期天候情報",
    "display_label": "高温に関する早期天候情報",
    "severity": 0.0,
    "status": "情報",
    "detail": "東海地方を対象とした高温に関する早期天候情報が発表されている。",
    "observed_at": "2026-05-25 05:09:51",
    "updated_at": "2026-05-26 11:45:31",
    "is_simulated": false,
    "created_at": "2026-05-26 11:45:31"
  },
  {
    "source": "愛知県 災害関連情報ポータル",
    "source_url": "https://www-bousai1.kenbousai-cloud.pref.aichi.jp/pub_web/portal-top/",
    "category": "その他",
    "area": "愛知県",
    "label": "発表中の警報・注意報なし",
    "display_label": "発表中の警報・注意報なし",
    "severity": 0.0,
    "status": "通常",
    "detail": "気象警報・注意報、地震、津波ともに発表中の情報はない。",
    "observed_at": "2026-05-26 10:35:01",
    "updated_at": "2026-05-26 11:45:31",
    "is_simulated": false,
    "created_at": "2026-05-26 11:45:31"
  },
  {
    "source": "名古屋鉄道 運行情報",
    "source_url": "https://top.meitetsu.co.jp/em/?mediacd=012",
    "category": "その他",
    "area": "愛知県",
    "label": "遅延なし",
    "display_label": "遅延なし",
    "severity": 0.0,
    "status": "通常",
    "detail": "15分以上の列車の遅れはございません。",
    "observed_at": "2026-05-26 11:44:39",
    "updated_at": "2026-05-26 11:45:31",
    "is_simulated": false,
    "created_at": "2026-05-26 11:45:31"
  },
  {
    "source": "リニモ 運行情報",
    "source_url": "https://www.linimo.jp//delay/",
    "category": "その他",
    "area": "愛知県・長久手市周辺",
    "label": "平常通り運行",
    "display_label": "平常通り運行",
    "severity": 0.0,
    "status": "通常",
    "detail": "現在、平常通り運行しております。",
    "observed_at": "2026-05-26 11:44:39",
    "updated_at": "2026-05-26 11:45:31",
    "is_simulated": false,
    "created_at": "2026-05-26 11:45:31"
  },
  {
    "source": "愛知環状鉄道 運行情報",
    "source_url": "https://www.aikanrailway.co.jp/train/",
    "category": "その他",
    "area": "愛知県・岡崎市から春日井市周辺",
    "label": "愛知環状鉄道 定刻どおり運転",
    "display_label": "愛知環状鉄道 定刻どおり運転",
    "severity": 0.0,
    "status": "通常",
    "detail": "11時43分現在、列車は定刻どおり運転しています。",
    "observed_at": "2026-05-26 11:43:58",
    "updated_at": "2026-05-26 11:45:31",
    "is_simulated": false,
    "created_at": "2026-05-26 11:45:31"
  },
  {
    "source": "名古屋市交通局 運行情報",
    "source_url": "https://www.kotsu.city.nagoya.jp/rp/emergency/",
    "category": "その他",
    "area": "名古屋市",
    "label": "市バス・地下鉄全線 平常運行",
    "display_label": "市バス・地下鉄全線 平常運行",
    "severity": 0.0,
    "status": "通常",
    "detail": "市バス、東山線、上飯田線、名城線、桜通線、鶴舞線ともに平常通り運行しています。",
    "observed_at": "2026-05-26 11:44:39",
    "updated_at": "2026-05-26 11:45:31",
    "is_simulated": false,
    "created_at": "2026-05-26 11:45:31"
  },
  {
    "source": "JR東海 運行情報",
    "source_url": "https://traininfo.jr-central.co.jp/zairaisen/",
    "category": "その他",
    "area": "愛知県・東海地方",
    "label": "高山線 杉原～猪谷間で運転見合わせ・ひだ列車運休",
    "display_label": "高山線 杉原～猪谷間で運転見合わせ・ひだ列車運休",
    "severity": 0.0,
    "status": "運休",
    "detail": "高山線杉原駅～猪谷駅間で線路設備確認のため運転見合わせ。ひだ列車（高山～富山間上下）で運休発生。",
    "observed_at": "2026-05-26 11:44:39",
    "updated_at": "2026-05-26 11:45:31",
    "is_simulated": false,
    "created_at": "2026-05-26 11:45:31"
  },
  {
    "source": "NEXCO中日本 交通情報",
    "source_url": "https://www.c-ihighway.jp/pcsite/map?area=area05",
    "category": "その他",
    "area": "愛知県・東海地方",
    "label": "東名高速 岩津バス停付近で事故・渋滞・規制",
    "display_label": "東名高速 岩津バス停付近で事故・渋滞・規制",
    "severity": 0.0,
    "status": "支障",
    "detail": "東名高速下り（大阪方面）岩津バス停付近で事故発生。また、東名高速各所で工事・作業による車線規制および渋滞（山北・蟹江・甲南・甲南トンネル等）が発生中。",
    "observed_at": "2026-05-26 11:42:54",
    "updated_at": "2026-05-26 11:45:31",
    "is_simulated": false,
    "created_at": "2026-05-26 11:45:31"
  },
  {
    "source": "気象庁防災情報XML",
    "source_url": "https://www.jma.go.jp/bosai/#pattern=default&area_type=offices&area_code=230000",
    "category": "その他",
    "area": "愛知県",
    "label": "高温に関する早期天候情報",
    "display_label": "高温に関する早期天候情報",
    "severity": 0.0,
    "status": "情報",
    "detail": "東海地方を対象に高温に関する早期天候情報が発表されている。",
    "observed_at": "2026-05-25 05:09:51",
    "updated_at": "2026-05-26 10:44:18",
    "is_simulated": false,
    "created_at": "2026-05-26 10:44:18"
  }
]
```

## Full Evidence DB Snapshot

```json
[
  {
    "source": "研究用模擬イベントデータ",
    "source_url": "simulation://urban-safety-research/airport_access/coastal-weather",
    "category": "気象",
    "area": "常滑市・知多半島沿岸",
    "label": "沿岸部の高波・強風に注意",
    "display_label": "【模擬データ】沿岸部の高波・強風に注意",
    "severity": 2.5,
    "status": "注意",
    "detail": "【模擬データ】沿岸部で高波と強風への注意が必要な想定です。 この情報は研究検証用であり、実際の公的発表ではありません。",
    "observed_at": "2026-05-29 05:01:11",
    "updated_at": "2026-05-29 05:01:11",
    "is_simulated": true,
    "created_at": "2026-05-29 05:01:11"
  },
  {
    "source": "研究用模擬イベントデータ",
    "source_url": "simulation://urban-safety-research/airport_access/airport-road-wind",
    "category": "道路",
    "area": "知多半島道路・空港連絡道路",
    "label": "強風に伴う速度規制",
    "display_label": "【模擬データ】強風に伴う速度規制",
    "severity": 2.8,
    "status": "注意",
    "detail": "【模擬データ】強風により空港連絡道路で速度規制が行われている想定です。 この情報は研究検証用であり、実際の公的発表ではありません。",
    "observed_at": "2026-05-29 05:01:11",
    "updated_at": "2026-05-29 05:01:11",
    "is_simulated": true,
    "created_at": "2026-05-29 05:01:11"
  },
  {
    "source": "研究用模擬イベントデータ",
    "source_url": "simulation://urban-safety-research/airport_access/airport-rail-delay",
    "category": "鉄道",
    "area": "名古屋市・常滑市・中部国際空港周辺",
    "label": "空港方面の鉄道に遅延",
    "display_label": "【模擬データ】空港方面の鉄道に遅延",
    "severity": 3.8,
    "status": "支障",
    "detail": "【模擬データ】中部国際空港方面へ向かう鉄道で遅延が発生している想定です。 この情報は研究検証用であり、実際の公的発表ではありません。",
    "observed_at": "2026-05-29 05:01:11",
    "updated_at": "2026-05-29 05:01:11",
    "is_simulated": true,
    "created_at": "2026-05-29 05:01:11"
  },
  {
    "source": "研究用模擬イベントデータ",
    "source_url": "simulation://urban-safety-research/evacuation/shelter-open",
    "category": "防災",
    "area": "名古屋市南部",
    "label": "避難所開設情報",
    "display_label": "【模擬データ】避難所開設情報",
    "severity": 4.0,
    "status": "警戒",
    "detail": "【模擬データ】複数の公共施設が避難所として開設された想定です。 この情報は研究検証用であり、実際の公的発表ではありません。",
    "observed_at": "2026-05-29 05:01:11",
    "updated_at": "2026-05-29 05:01:11",
    "is_simulated": true,
    "created_at": "2026-05-29 05:01:11"
  },
  {
    "source": "研究用模擬イベントデータ",
    "source_url": "simulation://urban-safety-research/evacuation/evacuation-order",
    "category": "防災",
    "area": "名古屋市南部",
    "label": "一部地域に避難指示",
    "display_label": "【模擬データ】一部地域に避難指示",
    "severity": 5.0,
    "status": "危険",
    "detail": "【模擬データ】河川増水により一部地域へ避難指示が発令された想定です。実際の避難判断には自治体の公的発表を確認してください。 この情報は研究検証用であり、実際の公的発表ではありません。",
    "observed_at": "2026-05-29 05:01:11",
    "updated_at": "2026-05-29 05:01:11",
    "is_simulated": true,
    "created_at": "2026-05-29 05:01:11"
  },
  {
    "source": "研究用模擬イベントデータ",
    "source_url": "simulation://urban-safety-research/weather_warning/strong-wind-warning",
    "category": "気象",
    "area": "知多半島・三河湾沿岸",
    "label": "暴風・高潮への警戒",
    "display_label": "【模擬データ】暴風・高潮への警戒",
    "severity": 4.2,
    "status": "警戒",
    "detail": "【模擬データ】沿岸部で強風と高潮の危険が高まっている想定です。 この情報は研究検証用であり、実際の公的発表ではありません。",
    "observed_at": "2026-05-29 05:01:11",
    "updated_at": "2026-05-29 05:01:11",
    "is_simulated": true,
    "created_at": "2026-05-29 05:01:11"
  },
  {
    "source": "研究用模擬イベントデータ",
    "source_url": "simulation://urban-safety-research/weather_warning/heavy-rain-warning",
    "category": "気象",
    "area": "愛知県西部",
    "label": "大雨警報・洪水警報",
    "display_label": "【模擬データ】大雨警報・洪水警報",
    "severity": 4.8,
    "status": "警戒",
    "detail": "【模擬データ】短時間強雨により低い土地の浸水、河川増水、交通機関の乱れが発生しやすい想定です。 この情報は研究検証用であり、実際の公的発表ではありません。",
    "observed_at": "2026-05-29 05:01:11",
    "updated_at": "2026-05-29 05:01:11",
    "is_simulated": true,
    "created_at": "2026-05-29 05:01:11"
  },
  {
    "source": "研究用模擬イベントデータ",
    "source_url": "simulation://urban-safety-research/road_closure/nagoya-expressway-congestion",
    "category": "道路",
    "area": "名古屋市",
    "label": "名古屋高速 渋滞・車線規制",
    "display_label": "【模擬データ】名古屋高速 渋滞・車線規制",
    "severity": 3.0,
    "status": "支障",
    "detail": "【模擬データ】工事と事故の影響により、都心環状線で渋滞と車線規制が発生している想定です。 この情報は研究検証用であり、実際の公的発表ではありません。",
    "observed_at": "2026-05-29 05:01:11",
    "updated_at": "2026-05-29 05:01:11",
    "is_simulated": true,
    "created_at": "2026-05-29 05:01:11"
  },
  {
    "source": "研究用模擬イベントデータ",
    "source_url": "simulation://urban-safety-research/road_closure/tomei-closure",
    "category": "道路",
    "area": "愛知県東部",
    "label": "東名高速 一部区間で通行止め",
    "display_label": "【模擬データ】東名高速 一部区間で通行止め",
    "severity": 5.0,
    "status": "危険",
    "detail": "【模擬データ】事故処理のため、豊川IC付近の上下線で通行止めが発生している想定です。 この情報は研究検証用であり、実際の公的発表ではありません。",
    "observed_at": "2026-05-29 05:01:11",
    "updated_at": "2026-05-29 05:01:11",
    "is_simulated": true,
    "created_at": "2026-05-29 05:01:11"
  },
  {
    "source": "研究用模擬イベントデータ",
    "source_url": "simulation://urban-safety-research/railway_disruption/meitetsu-delay",
    "category": "鉄道",
    "area": "名古屋市・常滑市",
    "label": "名鉄常滑線 遅延",
    "display_label": "【模擬データ】名鉄常滑線 遅延",
    "severity": 3.5,
    "status": "支障",
    "detail": "【模擬データ】空港方面へ向かう列車に遅延が発生している想定です。 この情報は研究検証用であり、実際の公的発表ではありません。",
    "observed_at": "2026-05-29 05:01:11",
    "updated_at": "2026-05-29 05:01:11",
    "is_simulated": true,
    "created_at": "2026-05-29 05:01:11"
  },
  {
    "source": "研究用模擬イベントデータ",
    "source_url": "simulation://urban-safety-research/railway_disruption/jr-tokaido-stop",
    "category": "鉄道",
    "area": "名古屋市・岡崎市・豊橋市",
    "label": "JR東海道線 一部区間で運転見合わせ",
    "display_label": "【模擬データ】JR東海道線 一部区間で運転見合わせ",
    "severity": 4.5,
    "status": "運休",
    "detail": "【模擬データ】大雨の影響により、名古屋から豊橋方面の一部区間で運転見合わせが発生している想定です。 この情報は研究検証用であり、実際の公的発表ではありません。",
    "observed_at": "2026-05-29 05:01:11",
    "updated_at": "2026-05-29 05:01:11",
    "is_simulated": true,
    "created_at": "2026-05-29 05:01:11"
  },
  {
    "source": "JR東海 運行情報",
    "source_url": "https://traininfo.jr-central.co.jp/zairaisen/",
    "category": "その他",
    "area": "愛知県・東海地方",
    "label": "高山線 杉原～猪谷間で運転見合わせ・ひだ列車運休",
    "display_label": "高山線 杉原～猪谷間で運転見合わせ・ひだ列車運休",
    "severity": 0.0,
    "status": "運休",
    "detail": "高山線杉原駅～猪谷駅間で線路設備確認のため運転見合わせ。ひだ列車（高山～富山間上下）で運休発生。",
    "observed_at": "2026-05-26 11:44:39",
    "updated_at": "2026-05-26 11:45:31",
    "is_simulated": false,
    "created_at": "2026-05-26 11:45:31"
  },
  {
    "source": "名古屋市交通局 運行情報",
    "source_url": "https://www.kotsu.city.nagoya.jp/rp/emergency/",
    "category": "その他",
    "area": "名古屋市",
    "label": "市バス・地下鉄全線 平常運行",
    "display_label": "市バス・地下鉄全線 平常運行",
    "severity": 0.0,
    "status": "通常",
    "detail": "市バス、東山線、上飯田線、名城線、桜通線、鶴舞線ともに平常通り運行しています。",
    "observed_at": "2026-05-26 11:44:39",
    "updated_at": "2026-05-26 11:45:31",
    "is_simulated": false,
    "created_at": "2026-05-26 11:45:31"
  },
  {
    "source": "リニモ 運行情報",
    "source_url": "https://www.linimo.jp//delay/",
    "category": "その他",
    "area": "愛知県・長久手市周辺",
    "label": "平常通り運行",
    "display_label": "平常通り運行",
    "severity": 0.0,
    "status": "通常",
    "detail": "現在、平常通り運行しております。",
    "observed_at": "2026-05-26 11:44:39",
    "updated_at": "2026-05-26 11:45:31",
    "is_simulated": false,
    "created_at": "2026-05-26 11:45:31"
  },
  {
    "source": "名古屋鉄道 運行情報",
    "source_url": "https://top.meitetsu.co.jp/em/?mediacd=012",
    "category": "その他",
    "area": "愛知県",
    "label": "遅延なし",
    "display_label": "遅延なし",
    "severity": 0.0,
    "status": "通常",
    "detail": "15分以上の列車の遅れはございません。",
    "observed_at": "2026-05-26 11:44:39",
    "updated_at": "2026-05-26 11:45:31",
    "is_simulated": false,
    "created_at": "2026-05-26 11:45:31"
  },
  {
    "source": "愛知環状鉄道 運行情報",
    "source_url": "https://www.aikanrailway.co.jp/train/",
    "category": "その他",
    "area": "愛知県・岡崎市から春日井市周辺",
    "label": "愛知環状鉄道 定刻どおり運転",
    "display_label": "愛知環状鉄道 定刻どおり運転",
    "severity": 0.0,
    "status": "通常",
    "detail": "11時43分現在、列車は定刻どおり運転しています。",
    "observed_at": "2026-05-26 11:43:58",
    "updated_at": "2026-05-26 11:45:31",
    "is_simulated": false,
    "created_at": "2026-05-26 11:45:31"
  },
  {
    "source": "NEXCO中日本 交通情報",
    "source_url": "https://www.c-ihighway.jp/pcsite/map?area=area05",
    "category": "その他",
    "area": "愛知県・東海地方",
    "label": "東名高速 岩津バス停付近で事故・渋滞・規制",
    "display_label": "東名高速 岩津バス停付近で事故・渋滞・規制",
    "severity": 0.0,
    "status": "支障",
    "detail": "東名高速下り（大阪方面）岩津バス停付近で事故発生。また、東名高速各所で工事・作業による車線規制および渋滞（山北・蟹江・甲南・甲南トンネル等）が発生中。",
    "observed_at": "2026-05-26 11:42:54",
    "updated_at": "2026-05-26 11:45:31",
    "is_simulated": false,
    "created_at": "2026-05-26 11:45:31"
  },
  {
    "source": "JR東海 運行情報",
    "source_url": "https://traininfo.jr-central.co.jp/zairaisen/",
    "category": "その他",
    "area": "愛知県・岐阜県・高山線",
    "label": "高山線 運転見合わせ・運休",
    "display_label": "高山線 運転見合わせ・運休",
    "severity": 0.0,
    "status": "運休",
    "detail": "高山線杉原駅～猪谷駅間で線路設備確認のため運転見合わせ。高山～富山間（下り）および富山～高山間（上り）のひだで運休発生。",
    "observed_at": "2026-05-26 10:43:18",
    "updated_at": "2026-05-26 10:44:18",
    "is_simulated": false,
    "created_at": "2026-05-26 10:44:18"
  },
  {
    "source": "名古屋市交通局 運行情報",
    "source_url": "https://www.kotsu.city.nagoya.jp/rp/emergency/",
    "category": "その他",
    "area": "名古屋市",
    "label": "市バス・地下鉄全線平常運行",
    "display_label": "市バス・地下鉄全線平常運行",
    "severity": 0.0,
    "status": "通常",
    "detail": "市バス、東山線、上飯田線、名城線、桜通線、鶴舞線ともに平常通り運行しています。",
    "observed_at": "2026-05-26 10:43:18",
    "updated_at": "2026-05-26 10:44:18",
    "is_simulated": false,
    "created_at": "2026-05-26 10:44:18"
  },
  {
    "source": "愛知環状鉄道 運行情報",
    "source_url": "https://www.aikanrailway.co.jp/train/",
    "category": "その他",
    "area": "愛知県・岡崎市から春日井市周辺",
    "label": "定刻どおり運転",
    "display_label": "定刻どおり運転",
    "severity": 0.0,
    "status": "通常",
    "detail": "10時43分現在、列車は定刻どおり運転しています。",
    "observed_at": "2026-05-26 10:43:17",
    "updated_at": "2026-05-26 10:44:18",
    "is_simulated": false,
    "created_at": "2026-05-26 10:44:18"
  },
  {
    "source": "リニモ 運行情報",
    "source_url": "https://www.linimo.jp//delay/",
    "category": "その他",
    "area": "愛知県・長久手市周辺",
    "label": "平常通り運行",
    "display_label": "平常通り運行",
    "severity": 0.0,
    "status": "通常",
    "detail": "現在、平常通り運行しております。",
    "observed_at": "2026-05-26 10:43:17",
    "updated_at": "2026-05-26 10:44:18",
    "is_simulated": false,
    "created_at": "2026-05-26 10:44:18"
  },
  {
    "source": "名古屋鉄道 運行情報",
    "source_url": "https://top.meitetsu.co.jp/em/?mediacd=012",
    "category": "その他",
    "area": "愛知県",
    "label": "遅延なし",
    "display_label": "遅延なし",
    "severity": 0.0,
    "status": "通常",
    "detail": "15分以上の列車の遅れはございません。",
    "observed_at": "2026-05-26 10:43:17",
    "updated_at": "2026-05-26 10:44:18",
    "is_simulated": false,
    "created_at": "2026-05-26 10:44:18"
  },
  {
    "source": "NEXCO中日本 交通情報",
    "source_url": "https://www.c-ihighway.jp/pcsite/map?area=area05",
    "category": "その他",
    "area": "愛知県・東海地方 高速道路",
    "label": "高速道路 渋滞・規制情報",
    "display_label": "高速道路 渋滞・規制情報",
    "severity": 0.0,
    "status": "支障",
    "detail": "愛知県内では東名阪道蟹江IC付近（下り4km）、伊勢湾岸道名港西大橋付近（上り6km）で渋滞。また、東名音羽蒲郡IC～豊川IC上り、名神一宮IC～小牧IC上り、新名神亀山西JCT～甲賀土山IC下りなどで規制中。",
    "observed_at": "2026-05-26 10:42:54",
    "updated_at": "2026-05-26 10:44:18",
    "is_simulated": false,
    "created_at": "2026-05-26 10:44:18"
  },
  {
    "source": "愛知県 災害関連情報ポータル",
    "source_url": "https://www-bousai1.kenbousai-cloud.pref.aichi.jp/pub_web/portal-top/",
    "category": "その他",
    "area": "愛知県",
    "label": "発表中の警報・注意報なし",
    "display_label": "発表中の警報・注意報なし",
    "severity": 0.0,
    "status": "通常",
    "detail": "気象警報・注意報、地震、津波ともに発表中の情報はない。",
    "observed_at": "2026-05-26 10:35:01",
    "updated_at": "2026-05-26 11:45:31",
    "is_simulated": false,
    "created_at": "2026-05-26 11:45:31"
  },
  {
    "source": "愛知県 災害関連情報ポータル",
    "source_url": "https://www-bousai1.kenbousai-cloud.pref.aichi.jp/pub_web/portal-top/",
    "category": "その他",
    "area": "愛知県",
    "label": "警報・注意報なし",
    "display_label": "警報・注意報なし",
    "severity": 0.0,
    "status": "通常",
    "detail": "発表中の気象警報・注意報は見つかりません。全市町村 maxLevel=0。",
    "observed_at": "2026-05-26 10:35:01",
    "updated_at": "2026-05-26 10:44:18",
    "is_simulated": false,
    "created_at": "2026-05-26 10:44:18"
  },
  {
    "source": "愛知県 災害関連情報ポータル",
    "source_url": "https://www-bousai1.kenbousai-cloud.pref.aichi.jp/pub_web/portal-top/",
    "category": "その他",
    "area": "愛知県",
    "label": "地震情報なし",
    "display_label": "地震情報なし",
    "severity": 0.0,
    "status": "通常",
    "detail": "現在、地震の対象地域は見つかりません。",
    "observed_at": "2026-05-26 09:22:00",
    "updated_at": "2026-05-26 10:44:18",
    "is_simulated": false,
    "created_at": "2026-05-26 10:44:18"
  },
  {
    "source": "気象庁防災情報XML",
    "source_url": "https://www.data.jma.go.jp/developer/xml/data/20260525050951_0_VPAW51_230000.xml",
    "category": "その他",
    "area": "愛知県",
    "label": "高温に関する早期天候情報",
    "display_label": "高温に関する早期天候情報",
    "severity": 0.0,
    "status": "情報",
    "detail": "東海地方を対象とした高温に関する早期天候情報が発表されている。",
    "observed_at": "2026-05-25 05:09:51",
    "updated_at": "2026-05-26 11:45:31",
    "is_simulated": false,
    "created_at": "2026-05-26 11:45:31"
  },
  {
    "source": "気象庁防災情報XML",
    "source_url": "https://www.jma.go.jp/bosai/#pattern=default&area_type=offices&area_code=230000",
    "category": "その他",
    "area": "愛知県",
    "label": "高温に関する早期天候情報",
    "display_label": "高温に関する早期天候情報",
    "severity": 0.0,
    "status": "情報",
    "detail": "東海地方を対象に高温に関する早期天候情報が発表されている。",
    "observed_at": "2026-05-25 05:09:51",
    "updated_at": "2026-05-26 10:44:18",
    "is_simulated": false,
    "created_at": "2026-05-26 10:44:18"
  },
  {
    "source": "JR東海 運行情報",
    "source_url": "https://traininfo.jr-central.co.jp/zairaisen/",
    "category": "その他",
    "area": "愛知県・東海地方",
    "label": "高山線運転見合わせ、東海道線一部遅れ・運休",
    "display_label": "高山線運転見合わせ、東海道線一部遅れ・運休",
    "severity": 0.0,
    "status": "支障",
    "detail": "高山線杉原駅～猪谷駅間で線路設備確認のため運転見合わせ。東海道線熱海駅・草薙駅で遅れ、豊橋駅～西小坂井駅間で自動車衝突により一時運休（15時30分再開）。飯田線船町駅で自動車衝突により一時運休（15時52分再開）。",
    "observed_at": "2026-05-24 16:26:11",
    "updated_at": "2026-05-24 16:27:08",
    "is_simulated": false,
    "created_at": "2026-05-24 16:27:08"
  },
  {
    "source": "名古屋市交通局 運行情報",
    "source_url": "https://www.kotsu.city.nagoya.jp/rp/emergency/",
    "category": "その他",
    "area": "名古屋市",
    "label": "市バス・地下鉄全線 平常運行",
    "display_label": "市バス・地下鉄全線 平常運行",
    "severity": 0.0,
    "status": "通常",
    "detail": "市バス、東山線、上飯田線、名城線、桜通線、鶴舞線ともに平常通り運行しています。",
    "observed_at": "2026-05-24 16:26:11",
    "updated_at": "2026-05-24 16:27:08",
    "is_simulated": false,
    "created_at": "2026-05-24 16:27:08"
  },
  {
    "source": "リニモ 運行情報",
    "source_url": "https://www.linimo.jp//delay/",
    "category": "その他",
    "area": "愛知県・長久手市周辺",
    "label": "リニモの平常運行",
    "display_label": "リニモの平常運行",
    "severity": 0.0,
    "status": "通常",
    "detail": "現在、平常通り運行しております。",
    "observed_at": "2026-05-24 16:26:10",
    "updated_at": "2026-05-24 16:27:08",
    "is_simulated": false,
    "created_at": "2026-05-24 16:27:08"
  },
  {
    "source": "名古屋鉄道 運行情報",
    "source_url": "https://top.meitetsu.co.jp/em/?mediacd=012",
    "category": "その他",
    "area": "愛知県（名古屋本線 豊橋駅～伊奈駅間）",
    "label": "名古屋本線 豊橋駅～伊奈駅間の運転再開",
    "display_label": "名古屋本線 豊橋駅～伊奈駅間の運転再開",
    "severity": 0.0,
    "status": "通常",
    "detail": "名古屋本線 豊橋駅～伊奈駅間は、16時03分に運転を再開しました。振替輸送は16時30分で終了しました。",
    "observed_at": "2026-05-24 16:26:10",
    "updated_at": "2026-05-24 16:27:08",
    "is_simulated": false,
    "created_at": "2026-05-24 16:27:08"
  },
  {
    "source": "愛知環状鉄道 運行情報",
    "source_url": "https://www.aikanrailway.co.jp/train/",
    "category": "その他",
    "area": "愛知県・岡崎市から春日井市周辺",
    "label": "愛知環状鉄道 定刻どおり運転",
    "display_label": "愛知環状鉄道 定刻どおり運転",
    "severity": 0.0,
    "status": "通常",
    "detail": "16時25分現在、列車は定刻どおり運転しています。",
    "observed_at": "2026-05-24 16:25:59",
    "updated_at": "2026-05-24 16:27:08",
    "is_simulated": false,
    "created_at": "2026-05-24 16:27:08"
  },
  {
    "source": "NEXCO中日本 交通情報",
    "source_url": "https://www.c-ihighway.jp/pcsite/map?area=area05",
    "category": "その他",
    "area": "愛知県・東海地方",
    "label": "新東名事故、東名・中央道等渋滞・規制",
    "display_label": "新東名事故、東名・中央道等渋滞・規制",
    "severity": 0.0,
    "status": "支障",
    "detail": "新東名浜松浜北IC付近で事故。東名（日進JCT～豊川IC等）、中央道（土岐トンネル等）、北陸道等で渋滞。東名、中央道、新東名等で工事による車線規制・入口出口規制実施中。",
    "observed_at": "2026-05-24 16:22:39",
    "updated_at": "2026-05-24 16:27:08",
    "is_simulated": false,
    "created_at": "2026-05-24 16:27:08"
  },
  {
    "source": "JR東海 運行情報",
    "source_url": "https://traininfo.jr-central.co.jp/zairaisen/",
    "category": "その他",
    "area": "愛知県・東海地方",
    "label": "高山線・飯田線 運休・遅れ、東海道線 遅れ",
    "display_label": "高山線・飯田線 運休・遅れ、東海道線 遅れ",
    "severity": 0.0,
    "status": "支障",
    "detail": "高山線杉原～猪谷間で運転見合わせ。飯田線豊橋～豊川間で運転見合わせ（再開要時間）。東海道線熱海～豊橋間で遅れ。東海道線豊橋～米原間で一時運休後15時30分に再開。",
    "observed_at": "2026-05-24 15:37:44",
    "updated_at": "2026-05-24 15:38:22",
    "is_simulated": false,
    "created_at": "2026-05-24 15:38:22"
  },
  {
    "source": "名古屋市交通局 運行情報",
    "source_url": "https://www.kotsu.city.nagoya.jp/rp/emergency/",
    "category": "その他",
    "area": "名古屋市",
    "label": "市バス・地下鉄全線 平常運行",
    "display_label": "市バス・地下鉄全線 平常運行",
    "severity": 0.0,
    "status": "通常",
    "detail": "市バス、東山線、上飯田線、名城線、桜通線、鶴舞線ともに平常通り運行しています。",
    "observed_at": "2026-05-24 15:37:43",
    "updated_at": "2026-05-24 15:38:22",
    "is_simulated": false,
    "created_at": "2026-05-24 15:38:22"
  },
  {
    "source": "名古屋鉄道 運行情報",
    "source_url": "https://top.meitetsu.co.jp/em/?mediacd=012",
    "category": "その他",
    "area": "愛知県",
    "label": "名古屋本線 豊橋駅～伊奈駅間で運転見合わせ・遅延",
    "display_label": "名古屋本線 豊橋駅～伊奈駅間で運転見合わせ・遅延",
    "severity": 0.0,
    "status": "支障",
    "detail": "名古屋本線 豊橋駅～伊奈駅間（JR共用区間）で橋桁に車が衝突した情報があり、運転見合わせおよび遅延が発生しています。振替輸送は実施していません。詳細はJR東海ホームページをご確認ください。",
    "observed_at": "2026-05-24 15:37:43",
    "updated_at": "2026-05-24 15:38:22",
    "is_simulated": false,
    "created_at": "2026-05-24 15:38:22"
  },
  {
    "source": "NEXCO中日本 交通情報",
    "source_url": "https://www.c-ihighway.jp/pcsite/map?area=area05",
    "category": "その他",
    "area": "愛知県・東海地方",
    "label": "高速道路 事故・渋滞・規制情報",
    "display_label": "高速道路 事故・渋滞・規制情報",
    "severity": 0.0,
    "status": "情報",
    "detail": "新東名浜松浜北IC付近で事故。東名高速で豊川IC付近、岩津バス停付近、大井川焼津藤枝スマートIC付近などで車線規制（工事）。中央道、上信越道、長野道で渋滞。",
    "observed_at": "2026-05-24 15:32:39",
    "updated_at": "2026-05-24 15:38:22",
    "is_simulated": false,
    "created_at": "2026-05-24 15:38:22"
  },
  {
    "source": "JR東海 運行情報",
    "source_url": "https://traininfo.jr-central.co.jp/zairaisen/",
    "category": "その他",
    "area": "愛知県・東海地方",
    "label": "高山線運転見合わせ・東海道線遅れ",
    "display_label": "高山線運転見合わせ・東海道線遅れ",
    "severity": 0.0,
    "status": "運休",
    "detail": "高山線杉原駅～猪谷駅間で線路設備確認のため運転見合わせ。東海道線(熱海～豊橋)で乗務員物音確認およびトラブルにより遅れ発生。",
    "observed_at": "2026-05-24 14:53:58",
    "updated_at": "2026-05-24 14:54:40",
    "is_simulated": false,
    "created_at": "2026-05-24 14:54:40"
  },
  {
    "source": "名古屋市交通局 運行情報",
    "source_url": "https://www.kotsu.city.nagoya.jp/rp/emergency/",
    "category": "その他",
    "area": "名古屋市",
    "label": "市バス・地下鉄全線 平常運行",
    "display_label": "市バス・地下鉄全線 平常運行",
    "severity": 0.0,
    "status": "通常",
    "detail": "市バス、東山線、上飯田線、名城線、桜通線、鶴舞線が平常通り運行しています。",
    "observed_at": "2026-05-24 14:53:58",
    "updated_at": "2026-05-24 14:54:40",
    "is_simulated": false,
    "created_at": "2026-05-24 14:54:40"
  },
  {
    "source": "名古屋鉄道 運行情報",
    "source_url": "https://top.meitetsu.co.jp/em/?mediacd=012",
    "category": "その他",
    "area": "愛知県",
    "label": "名鉄 15分以上の遅れなし",
    "display_label": "名鉄 15分以上の遅れなし",
    "severity": 0.0,
    "status": "通常",
    "detail": "15分以上の列車の遅れはございません。",
    "observed_at": "2026-05-24 14:53:58",
    "updated_at": "2026-05-24 14:54:40",
    "is_simulated": false,
    "created_at": "2026-05-24 14:54:40"
  },
  {
    "source": "NEXCO中日本 交通情報",
    "source_url": "https://www.c-ihighway.jp/pcsite/map?area=area05",
    "category": "その他",
    "area": "愛知県・東海地方",
    "label": "高速道路工事規制・渋滞情報",
    "display_label": "高速道路工事規制・渋滞情報",
    "severity": 0.0,
    "status": "支障",
    "detail": "東名高速、新東名高速、名神高速、中央道などで工事による車線規制・入口出口規制を実施。長野道岡谷JCT付近で渋滞(4km)発生。",
    "observed_at": "2026-05-24 14:52:44",
    "updated_at": "2026-05-24 14:54:40",
    "is_simulated": false,
    "created_at": "2026-05-24 14:54:40"
  },
  {
    "source": "愛知県 災害関連情報ポータル",
    "source_url": "https://www-bousai1.kenbousai-cloud.pref.aichi.jp/pub_web/portal-top/",
    "category": "その他",
    "area": "愛知県",
    "label": "愛知県 波浪注意報（常滑市・南知多町・美浜町・田原市）",
    "display_label": "愛知県 波浪注意報（常滑市・南知多町・美浜町・田原市）",
    "severity": 0.0,
    "status": "注意",
    "detail": "常滑市、南知多町、美浜町、田原市で波浪注意報が発表中。",
    "observed_at": "2026-05-24 10:04:00",
    "updated_at": "2026-05-24 16:27:08",
    "is_simulated": false,
    "created_at": "2026-05-24 16:27:08"
  },
  {
    "source": "気象庁防災情報XML",
    "source_url": "https://www.jma.go.jp/bosai/#pattern=default&area_type=offices&area_code=230000",
    "category": "その他",
    "area": "愛知県",
    "label": "愛知県 波浪注意報",
    "display_label": "愛知県 波浪注意報",
    "severity": 0.0,
    "status": "注意",
    "detail": "愛知県では、２４日夜遅くまで高波に注意してください。",
    "observed_at": "2026-05-24 01:04:17",
    "updated_at": "2026-05-24 16:27:08",
    "is_simulated": false,
    "created_at": "2026-05-24 16:27:08"
  },
  {
    "source": "愛知県 災害関連情報ポータル",
    "source_url": "https://www-bousai1.kenbousai-cloud.pref.aichi.jp/pub_web/portal-top/",
    "category": "その他",
    "area": "愛知県",
    "label": "津波情報なし",
    "display_label": "津波情報なし",
    "severity": 0.0,
    "status": "通常",
    "detail": "現在、津波の対象地域は見つかりません。",
    "observed_at": "2026-04-20 23:47:01",
    "updated_at": "2026-05-26 10:44:18",
    "is_simulated": false,
    "created_at": "2026-05-26 10:44:18"
  }
]
```

## Generator Prompt

```text
あなたは都市安全情報を整理する研究用アシスタントです。
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


ユーザー質問:
中部国際空港へ向かう時に注意すべき情報は？

都市安全情報DB:
- 研究用模擬イベントデータ / 道路 / 愛知県東部 / 東名高速 一部区間で通行止め / status=危険 / severity=5.0 / observed_at=2026-05-29 05:01:11 / updated_at=2026-05-29 05:01:11 / url=simulation://urban-safety-research/road_closure/tomei-closure / / 模擬データ / detail=【模擬データ】事故処理のため、豊川IC付近の上下線で通行止めが発生している想定です。 この情報は研究検証用であり、実際の公的発表ではありません。
- 研究用模擬イベントデータ / 防災 / 名古屋市南部 / 一部地域に避難指示 / status=危険 / severity=5.0 / observed_at=2026-05-29 05:01:11 / updated_at=2026-05-29 05:01:11 / url=simulation://urban-safety-research/evacuation/evacuation-order / / 模擬データ / detail=【模擬データ】河川増水により一部地域へ避難指示が発令された想定です。実際の避難判断には自治体の公的発表を確認してください。 この情報は研究検証用であり、実際の公的発表ではありません。
- 研究用模擬イベントデータ / 気象 / 愛知県西部 / 大雨警報・洪水警報 / status=警戒 / severity=4.8 / observed_at=2026-05-29 05:01:11 / updated_at=2026-05-29 05:01:11 / url=simulation://urban-safety-research/weather_warning/heavy-rain-warning / / 模擬データ / detail=【模擬データ】短時間強雨により低い土地の浸水、河川増水、交通機関の乱れが発生しやすい想定です。 この情報は研究検証用であり、実際の公的発表ではありません。
- 研究用模擬イベントデータ / 鉄道 / 名古屋市・岡崎市・豊橋市 / JR東海道線 一部区間で運転見合わせ / status=運休 / severity=4.5 / observed_at=2026-05-29 05:01:11 / updated_at=2026-05-29 05:01:11 / url=simulation://urban-safety-research/railway_disruption/jr-tokaido-stop / / 模擬データ / detail=【模擬データ】大雨の影響により、名古屋から豊橋方面の一部区間で運転見合わせが発生している想定です。 この情報は研究検証用であり、実際の公的発表ではありません。
- 研究用模擬イベントデータ / 気象 / 知多半島・三河湾沿岸 / 暴風・高潮への警戒 / status=警戒 / severity=4.2 / observed_at=2026-05-29 05:01:11 / updated_at=2026-05-29 05:01:11 / url=simulation://urban-safety-research/weather_warning/strong-wind-warning / / 模擬データ / detail=【模擬データ】沿岸部で強風と高潮の危険が高まっている想定です。 この情報は研究検証用であり、実際の公的発表ではありません。
- 研究用模擬イベントデータ / 防災 / 名古屋市南部 / 避難所開設情報 / status=警戒 / severity=4.0 / observed_at=2026-05-29 05:01:11 / updated_at=2026-05-29 05:01:11 / url=simulation://urban-safety-research/evacuation/shelter-open / / 模擬データ / detail=【模擬データ】複数の公共施設が避難所として開設された想定です。 この情報は研究検証用であり、実際の公的発表ではありません。
- 研究用模擬イベントデータ / 鉄道 / 名古屋市・常滑市・中部国際空港周辺 / 空港方面の鉄道に遅延 / status=支障 / severity=3.8 / observed_at=2026-05-29 05:01:11 / updated_at=2026-05-29 05:01:11 / url=simulation://urban-safety-research/airport_access/airport-rail-delay / / 模擬データ / detail=【模擬データ】中部国際空港方面へ向かう鉄道で遅延が発生している想定です。 この情報は研究検証用であり、実際の公的発表ではありません。
- 研究用模擬イベントデータ / 鉄道 / 名古屋市・常滑市 / 名鉄常滑線 遅延 / status=支障 / severity=3.5 / observed_at=2026-05-29 05:01:11 / updated_at=2026-05-29 05:01:11 / url=simulation://urban-safety-research/railway_disruption/meitetsu-delay / / 模擬データ / detail=【模擬データ】空港方面へ向かう列車に遅延が発生している想定です。 この情報は研究検証用であり、実際の公的発表ではありません。
- 研究用模擬イベントデータ / 道路 / 名古屋市 / 名古屋高速 渋滞・車線規制 / status=支障 / severity=3.0 / observed_at=2026-05-29 05:01:11 / updated_at=2026-05-29 05:01:11 / url=simulation://urban-safety-research/road_closure/nagoya-expressway-congestion / / 模擬データ / detail=【模擬データ】工事と事故の影響により、都心環状線で渋滞と車線規制が発生している想定です。 この情報は研究検証用であり、実際の公的発表ではありません。
- 研究用模擬イベントデータ / 道路 / 知多半島道路・空港連絡道路 / 強風に伴う速度規制 / status=注意 / severity=2.8 / observed_at=2026-05-29 05:01:11 / updated_at=2026-05-29 05:01:11 / url=simulation://urban-safety-research/airport_access/airport-road-wind / / 模擬データ / detail=【模擬データ】強風により空港連絡道路で速度規制が行われている想定です。 この情報は研究検証用であり、実際の公的発表ではありません。
- 研究用模擬イベントデータ / 気象 / 常滑市・知多半島沿岸 / 沿岸部の高波・強風に注意 / status=注意 / severity=2.5 / observed_at=2026-05-29 05:01:11 / updated_at=2026-05-29 05:01:11 / url=simulation://urban-safety-research/airport_access/coastal-weather / / 模擬データ / detail=【模擬データ】沿岸部で高波と強風への注意が必要な想定です。 この情報は研究検証用であり、実際の公的発表ではありません。
- 気象庁防災情報XML / その他 / 愛知県 / 高温に関する早期天候情報 / status=情報 / severity=0.0 / observed_at=2026-05-25 05:09:51 / updated_at=2026-05-26 11:45:31 / url=https://www.data.jma.go.jp/developer/xml/data/20260525050951_0_VPAW51_230000.xml / / 公的情報 / detail=東海地方を対象とした高温に関する早期天候情報が発表されている。
- 愛知県 災害関連情報ポータル / その他 / 愛知県 / 発表中の警報・注意報なし / status=通常 / severity=0.0 / observed_at=2026-05-26 10:35:01 / updated_at=2026-05-26 11:45:31 / url=https://www-bousai1.kenbousai-cloud.pref.aichi.jp/pub_web/portal-top/ / / 公的情報 / detail=気象警報・注意報、地震、津波ともに発表中の情報はない。
- 名古屋鉄道 運行情報 / その他 / 愛知県 / 遅延なし / status=通常 / severity=0.0 / observed_at=2026-05-26 11:44:39 / updated_at=2026-05-26 11:45:31 / url=https://top.meitetsu.co.jp/em/?mediacd=012 / / 公的情報 / detail=15分以上の列車の遅れはございません。
- リニモ 運行情報 / その他 / 愛知県・長久手市周辺 / 平常通り運行 / status=通常 / severity=0.0 / observed_at=2026-05-26 11:44:39 / updated_at=2026-05-26 11:45:31 / url=https://www.linimo.jp//delay/ / / 公的情報 / detail=現在、平常通り運行しております。
- 愛知環状鉄道 運行情報 / その他 / 愛知県・岡崎市から春日井市周辺 / 愛知環状鉄道 定刻どおり運転 / status=通常 / severity=0.0 / observed_at=2026-05-26 11:43:58 / updated_at=2026-05-26 11:45:31 / url=https://www.aikanrailway.co.jp/train/ / / 公的情報 / detail=11時43分現在、列車は定刻どおり運転しています。
- 名古屋市交通局 運行情報 / その他 / 名古屋市 / 市バス・地下鉄全線 平常運行 / status=通常 / severity=0.0 / observed_at=2026-05-26 11:44:39 / updated_at=2026-05-26 11:45:31 / url=https://www.kotsu.city.nagoya.jp/rp/emergency/ / / 公的情報 / detail=市バス、東山線、上飯田線、名城線、桜通線、鶴舞線ともに平常通り運行しています。
- JR東海 運行情報 / その他 / 愛知県・東海地方 / 高山線 杉原～猪谷間で運転見合わせ・ひだ列車運休 / status=運休 / severity=0.0 / observed_at=2026-05-26 11:44:39 / updated_at=2026-05-26 11:45:31 / url=https://traininfo.jr-central.co.jp/zairaisen/ / / 公的情報 / detail=高山線杉原駅～猪谷駅間で線路設備確認のため運転見合わせ。ひだ列車（高山～富山間上下）で運休発生。
- NEXCO中日本 交通情報 / その他 / 愛知県・東海地方 / 東名高速 岩津バス停付近で事故・渋滞・規制 / status=支障 / severity=0.0 / observed_at=2026-05-26 11:42:54 / updated_at=2026-05-26 11:45:31 / url=https://www.c-ihighway.jp/pcsite/map?area=area05 / / 公的情報 / detail=東名高速下り（大阪方面）岩津バス停付近で事故発生。また、東名高速各所で工事・作業による車線規制および渋滞（山北・蟹江・甲南・甲南トンネル等）が発生中。
- 気象庁防災情報XML / その他 / 愛知県 / 高温に関する早期天候情報 / status=情報 / severity=0.0 / observed_at=2026-05-25 05:09:51 / updated_at=2026-05-26 10:44:18 / url=https://www.jma.go.jp/bosai/#pattern=default&area_type=offices&area_code=230000 / / 公的情報 / detail=東海地方を対象に高温に関する早期天候情報が発表されている。

```

## Verifier Prompt

```text
Verifier Agent は本研究テーマから外しているため使用しません。
```

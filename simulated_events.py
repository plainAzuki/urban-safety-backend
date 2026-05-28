"""卒業研究の異常時検証に使う模擬イベントデータ。

ここで作るデータは公的情報ではない。API と UI では必ず is_simulated=True
および「模擬データ」ラベルを表示し、実際の防災・交通判断に使わない。
"""

from datetime import datetime
from typing import Optional


SIMULATED_SOURCE = "研究用模擬イベントデータ"
SIMULATED_SOURCE_URL = "simulation://urban-safety-research"


SCENARIOS = {
    "normal_sample": {
        "name": "通常時サンプル",
        "description": "公式情報が大きな異常を示していない日常時の比較用データ。",
    },
    "railway_disruption": {
        "name": "鉄道運転見合わせ",
        "description": "JR・名鉄・地下鉄などの障害表示を確認するための模擬シナリオ。",
    },
    "road_closure": {
        "name": "高速道路事故・通行止め",
        "description": "道路交通カテゴリの支障・危険表示を確認するための模擬シナリオ。",
    },
    "weather_warning": {
        "name": "大雨・暴風警報",
        "description": "気象警報とリスク提示を確認するための模擬シナリオ。",
    },
    "evacuation": {
        "name": "避難情報",
        "description": "防災カテゴリの高リスク表示と出典注記を確認するための模擬シナリオ。",
    },
    "airport_access": {
        "name": "空港アクセス注意",
        "description": "空港・駅・主要道路へ向かう際の複合的な注意情報を確認するシナリオ。",
    },
    "multi_event": {
        "name": "複数イベント同時発生",
        "description": "鉄道・道路・気象・防災が同時に発生する異常時デモ用シナリオ。",
    },
}


def now_text() -> str:
    """模擬イベントの発生・更新時刻を統一形式で返す。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def simulated_event(
    scenario: str,
    event_id: str,
    category: str,
    area: str,
    label: str,
    status: str,
    severity: float,
    detail: str,
    display_label: Optional[str] = None,
) -> dict:
    """Evidence DB に保存できる模擬イベント1件を作る。"""
    timestamp = now_text()
    return {
        "id": f"simulation:{scenario}:{event_id}",
        "source": SIMULATED_SOURCE,
        "source_url": f"{SIMULATED_SOURCE_URL}/{scenario}/{event_id}",
        "category": category,
        "area": area,
        "label": label,
        "display_label": display_label or f"【模擬データ】{label}",
        "severity": severity,
        "status": status,
        "detail": f"【模擬データ】{detail} この情報は研究検証用であり、実際の公的発表ではありません。",
        "observed_at": timestamp,
        "updated_at": timestamp,
        "is_simulated": True,
    }


def normal_sample_events() -> list[dict]:
    """通常時比較用の低リスク模擬データ。"""
    return [
        simulated_event(
            "normal_sample",
            "railway-normal",
            "鉄道",
            "名古屋市・愛知県内主要路線",
            "主要鉄道路線 平常運行",
            "通常",
            0.0,
            "JR・名鉄・地下鉄の大きな運行障害がない状態を想定した比較用データです。",
        ),
        simulated_event(
            "normal_sample",
            "weather-info",
            "気象",
            "愛知県",
            "気象に関する特段の警報なし",
            "情報",
            0.5,
            "通常時の概要表示と異常時表示の差分を確認するための比較用データです。",
        ),
    ]


def railway_disruption_events() -> list[dict]:
    """鉄道障害の模擬データ。"""
    return [
        simulated_event(
            "railway_disruption",
            "jr-tokaido-stop",
            "鉄道",
            "名古屋市・岡崎市・豊橋市",
            "JR東海道線 一部区間で運転見合わせ",
            "運休",
            4.5,
            "大雨の影響により、名古屋から豊橋方面の一部区間で運転見合わせが発生している想定です。",
        ),
        simulated_event(
            "railway_disruption",
            "meitetsu-delay",
            "鉄道",
            "名古屋市・常滑市",
            "名鉄常滑線 遅延",
            "支障",
            3.5,
            "空港方面へ向かう列車に遅延が発生している想定です。",
        ),
    ]


def road_closure_events() -> list[dict]:
    """道路障害の模擬データ。"""
    return [
        simulated_event(
            "road_closure",
            "tomei-closure",
            "道路",
            "愛知県東部",
            "東名高速 一部区間で通行止め",
            "危険",
            5.0,
            "事故処理のため、豊川IC付近の上下線で通行止めが発生している想定です。",
        ),
        simulated_event(
            "road_closure",
            "nagoya-expressway-congestion",
            "道路",
            "名古屋市",
            "名古屋高速 渋滞・車線規制",
            "支障",
            3.0,
            "工事と事故の影響により、都心環状線で渋滞と車線規制が発生している想定です。",
        ),
    ]


def weather_warning_events() -> list[dict]:
    """気象警報の模擬データ。"""
    return [
        simulated_event(
            "weather_warning",
            "heavy-rain-warning",
            "気象",
            "愛知県西部",
            "大雨警報・洪水警報",
            "警戒",
            4.8,
            "短時間強雨により低い土地の浸水、河川増水、交通機関の乱れが発生しやすい想定です。",
        ),
        simulated_event(
            "weather_warning",
            "strong-wind-warning",
            "気象",
            "知多半島・三河湾沿岸",
            "暴風・高潮への警戒",
            "警戒",
            4.2,
            "沿岸部で強風と高潮の危険が高まっている想定です。",
        ),
    ]


def evacuation_events() -> list[dict]:
    """避難情報の模擬データ。"""
    return [
        simulated_event(
            "evacuation",
            "evacuation-order",
            "防災",
            "名古屋市南部",
            "一部地域に避難指示",
            "危険",
            5.0,
            "河川増水により一部地域へ避難指示が発令された想定です。実際の避難判断には自治体の公的発表を確認してください。",
        ),
        simulated_event(
            "evacuation",
            "shelter-open",
            "防災",
            "名古屋市南部",
            "避難所開設情報",
            "警戒",
            4.0,
            "複数の公共施設が避難所として開設された想定です。",
        ),
    ]


def airport_access_events() -> list[dict]:
    """空港アクセス注意の模擬データ。"""
    return [
        simulated_event(
            "airport_access",
            "airport-rail-delay",
            "鉄道",
            "名古屋市・常滑市・中部国際空港周辺",
            "空港方面の鉄道に遅延",
            "支障",
            3.8,
            "中部国際空港方面へ向かう鉄道で遅延が発生している想定です。",
        ),
        simulated_event(
            "airport_access",
            "airport-road-wind",
            "道路",
            "知多半島道路・空港連絡道路",
            "強風に伴う速度規制",
            "注意",
            2.8,
            "強風により空港連絡道路で速度規制が行われている想定です。",
        ),
        simulated_event(
            "airport_access",
            "coastal-weather",
            "気象",
            "常滑市・知多半島沿岸",
            "沿岸部の高波・強風に注意",
            "注意",
            2.5,
            "沿岸部で高波と強風への注意が必要な想定です。",
        ),
    ]


def multi_event_events() -> list[dict]:
    """複数カテゴリ同時発生の模擬データ。"""
    events = []
    events.extend(railway_disruption_events())
    events.extend(road_closure_events())
    events.extend(weather_warning_events())
    events.extend(evacuation_events())
    events.extend(airport_access_events())
    return events


SCENARIO_BUILDERS = {
    "normal_sample": normal_sample_events,
    "railway_disruption": railway_disruption_events,
    "road_closure": road_closure_events,
    "weather_warning": weather_warning_events,
    "evacuation": evacuation_events,
    "airport_access": airport_access_events,
    "multi_event": multi_event_events,
}


def scenario_catalog() -> list[dict]:
    """APIで返す模擬シナリオ一覧。"""
    return [{"key": key, **value} for key, value in SCENARIOS.items()]


def build_simulated_events(scenario: str = "multi_event") -> list[dict]:
    """指定シナリオの模擬イベントを返す。"""
    if scenario not in SCENARIO_BUILDERS:
        valid = ", ".join(SCENARIO_BUILDERS)
        raise ValueError(f"unknown scenario: {scenario}. valid scenarios: {valid}")
    return SCENARIO_BUILDERS[scenario]()

"""公式情報をアプリ内の共通形式へ変換する処理。"""

from datetime import datetime


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_score(value: float) -> float:
    return round(max(0.0, min(float(value), 1.0)), 3)


def signals_from_event(event: dict) -> list[dict]:
    """現在はDB内の模擬値から公式API相当の信号を作る。実API接続時も返却形式は維持する。"""
    signals = []
    weather_severity = normalize_score(event.get("weather_severity", 0.0))
    transport_severity = normalize_score(event.get("transport_severity", 0.0))

    if weather_severity >= 0.6:
        signals.append({
            "source": "気象庁・自治体防災情報",
            "label": "大雨・河川リスク",
            "severity": weather_severity,
            "status": "warning" if weather_severity >= 0.8 else "watch",
            "detail": "大雨・河川増水に関連する公式情報を想定した信号です。",
            "observed_at": event.get("timestamp") or now_text(),
        })

    if transport_severity >= 0.6:
        is_railway = event.get("category") == "railway"
        signals.append({
            "source": "鉄道運行情報" if is_railway else "道路交通情報",
            "label": "鉄道運行リスク" if is_railway else "道路交通リスク",
            "severity": transport_severity,
            "status": "suspended" if is_railway and transport_severity >= 0.8 else "disrupted",
            "detail": "愛知県内の交通・鉄道APIから得る情報を想定した信号です。",
            "observed_at": event.get("timestamp") or now_text(),
        })

    return signals


def official_summary(signals: list[dict]) -> dict:
    if not signals:
        return {"max_severity": 0.0, "sources": [], "status": "normal"}
    max_severity = max(signal["severity"] for signal in signals)
    sources = sorted({signal["source"] for signal in signals})
    status = "warning" if max_severity >= 0.8 else "watch"
    return {"max_severity": max_severity, "sources": sources, "status": status}

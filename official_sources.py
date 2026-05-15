"""公式情報をアプリ内の共通形式へ変換する処理。"""

import os
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Optional

import httpx

JMA_FEED_URL = os.getenv("JMA_FEED_URL", "https://www.data.jma.go.jp/developer/xml/feed/extra_l.xml")
AICHI_KEYWORDS = ("愛知", "名古屋", "尾張", "西三河", "東三河", "豊田", "岡崎")
AICHI_AREA_CODE = "230000"

OFFICIAL_SOURCE_CATALOG = [
    {
        "key": "weather",
        "name": "気象庁・自治体防災情報",
        "area": "愛知県",
        "signal": "大雨・河川・浸水に関する注意信号",
        "status": "live_feed_ready",
    },
    {
        "key": "railway",
        "name": "鉄道運行情報",
        "area": "愛知県内の主要路線",
        "signal": "運転見合わせ・遅延に関する注意信号",
        "status": "mock_adapter",
    },
    {
        "key": "road",
        "name": "道路交通情報",
        "area": "愛知県内の主要道路",
        "signal": "事故・通行止め・混雑に関する注意信号",
        "status": "mock_adapter",
    },
]

# 気象庁XMLなどの公式発表を0〜1のseverityへ変換する固定表。
# 卒論では、この表を「公式信号を数値化する根拠」としてそのまま説明できる。
SEVERITY_CONVERSION_TABLE = [
    {"keyword": "解除", "severity": 0.0, "status": "normal", "reason": "公式にリスク低下が示されたため"},
    {"keyword": "特別警報", "severity": 1.0, "status": "warning", "reason": "最大級の危険度として扱うため"},
    {"keyword": "警報", "severity": 0.7, "status": "warning", "reason": "避難・移動判断に強く影響するため"},
    {"keyword": "土砂災害", "severity": 0.8, "status": "warning", "reason": "短時間で被害が拡大しやすいため"},
    {"keyword": "記録的短時間大雨", "severity": 0.8, "status": "warning", "reason": "局地的な浸水危険が高いため"},
    {"keyword": "竜巻", "severity": 0.7, "status": "watch", "reason": "発生可能性が高く時間依存性が強いため"},
    {"keyword": "注意報", "severity": 0.3, "status": "watch", "reason": "低〜中リスクの注意信号として扱うため"},
    {"keyword": "大雨", "severity": 0.6, "status": "watch", "reason": "浸水・河川増水と関係しやすいため"},
    {"keyword": "洪水", "severity": 0.6, "status": "watch", "reason": "河川・低地リスクと関係しやすいため"},
]


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_score(value: float) -> float:
    return round(max(0.0, min(float(value), 1.0)), 3)


def source_catalog() -> list[dict]:
    """実APIへ差し替える対象を、説明用にも使える一覧として返す。"""
    return OFFICIAL_SOURCE_CATALOG


def severity_conversion_table() -> list[dict]:
    """公式発表の種類とseverityの対応表を返す。"""
    return SEVERITY_CONVERSION_TABLE


def severity_from_conversion_table(text: str) -> Optional[tuple[float, str]]:
    for rule in SEVERITY_CONVERSION_TABLE:
        if rule["keyword"] in text:
            return rule["severity"], rule["status"]
    return None


def jma_severity_from_title(title: str) -> float:
    """気象庁XMLのタイトルから、アプリ共通の0〜1スコアへ粗く変換する。"""
    if title in {"気象特別警報・警報・注意報", "気象警報・注意報（Ｈ２７）"}:
        return 0.65
    converted = severity_from_conversion_table(title)
    if converted:
        return converted[0]
    return 0.4


def jma_severity_from_text(text: str, fallback_title: str) -> float:
    """本文が取れる場合は、電文種別名より本文の注意・警報表現を優先する。"""
    converted = severity_from_conversion_table(text)
    if converted:
        return converted[0]
    if "注意してください" in text:
        return 0.3
    return jma_severity_from_title(fallback_title)


def jma_status_from_severity(severity: float) -> str:
    if severity >= 0.7:
        return "warning"
    if severity >= 0.3:
        return "watch"
    return "info"


def local_name(tag: str) -> str:
    return tag.split("}", 1)[-1]


def first_text_by_tag(root: ET.Element, tag_name: str) -> str:
    for elem in root.iter():
        if local_name(elem.tag) == tag_name and elem.text and elem.text.strip():
            return elem.text.strip()
    return ""


def all_texts_by_tag(root: ET.Element, tag_name: str) -> list[str]:
    return [
        elem.text.strip()
        for elem in root.iter()
        if local_name(elem.tag) == tag_name and elem.text and elem.text.strip()
    ]


def parse_jma_detail(xml_bytes: bytes, fallback_title: str, href: str) -> dict:
    """JMA XML本文から、アプリ表示に使う見出しと状態を取り出す。"""
    root = ET.fromstring(xml_bytes)
    titles = all_texts_by_tag(root, "Title")
    head_title = next((title for title in titles if "愛知" in title), titles[-1] if titles else fallback_title)
    text = first_text_by_tag(root, "Text")
    severity = jma_severity_from_text(text, head_title)
    status = "normal" if severity < 0.4 else jma_status_from_severity(severity)
    detail = f"{text} / {href}" if text else href
    return {"label": head_title, "severity": severity, "status": status, "detail": detail}


async def fetch_jma_aichi_signals(limit: int = 20) -> list[dict]:
    """気象庁防災情報XMLのAtomフィードから、愛知県関連の見出しを取得する。"""
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(JMA_FEED_URL)
        resp.raise_for_status()

        root = ET.fromstring(resp.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        signals = []
        for entry in root.findall("atom:entry", ns):
            title = entry.findtext("atom:title", default="", namespaces=ns)
            updated = entry.findtext("atom:updated", default=now_text(), namespaces=ns)
            link = entry.find("atom:link", ns)
            href = link.get("href") if link is not None else JMA_FEED_URL
            is_aichi = AICHI_AREA_CODE in href or any(keyword in title for keyword in AICHI_KEYWORDS)
            if not is_aichi:
                continue
            metadata = {"label": title, "severity": jma_severity_from_title(title), "status": "", "detail": href}
            try:
                detail_resp = await client.get(href)
                detail_resp.raise_for_status()
                metadata = parse_jma_detail(detail_resp.content, title, href)
            except Exception:
                metadata["status"] = jma_status_from_severity(metadata["severity"])
            signals.append({
                "source": "気象庁防災情報XML",
                "area": "愛知県",
                "label": metadata["label"],
                "severity": metadata["severity"],
                "status": metadata["status"],
                "detail": metadata["detail"],
                "observed_at": updated,
            })
            if len(signals) >= limit:
                break
    return signals


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

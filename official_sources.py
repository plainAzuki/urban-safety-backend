"""公式ソースの定義及び生データの取得
LLM による正規化は official_service.pyで行う。
"""

import asyncio
import os
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from html import unescape
import httpx


JMA_FEED_URL = os.getenv("JMA_FEED_URL", "https://www.data.jma.go.jp/developer/xml/feed/extra_l.xml")
JMA_SOURCE_NAME = "気象庁防災情報XML"
JMA_AICHI_AREA = "愛知県"
JMA_AICHI_AUTHOR = "名古屋地方気象台"

MAX_OFFICIAL_RAW_CHARS = int(os.getenv("MAX_OFFICIAL_RAW_CHARS", "5000"))
HTTP_TIMEOUT_SECONDS = 20.0
USER_AGENT = "urban-safety-backend/3.0 official-info-check"
AICHI_BOUSAI_SOURCE_NAME = "愛知県 災害関連情報ポータル"
KOTSU_CITY_SOURCE_NAME = "名古屋市交通局 運行情報"
JR_TOKAI_SOURCE_NAME = "JR東海 運行情報"
IHIGHWAY_SOURCE_NAME = "NEXCO中日本 交通情報"
LINIMO_SOURCE_NAME = "リニモ 運行情報"
AIKAN_SOURCE_NAME = "愛知環状鉄道 運行情報"

AICHI_BOUSAI_URLS = {
    "weather_warn": "https://www-bousai1.kenbousai-cloud.pref.aichi.jp/data/weather/warn/top_warn.json",
    "quake": "https://www-bousai1.kenbousai-cloud.pref.aichi.jp/data/weather/quake/top_quake.json",
    "tsunami": "https://www-bousai1.kenbousai-cloud.pref.aichi.jp/data/weather/tsunami/top_tsunami.json",
    "disaster": "https://www-bousai1.kenbousai-cloud.pref.aichi.jp/data/disaster/top_disaster.json",
    "alert": "https://www-bousai1.kenbousai-cloud.pref.aichi.jp/data/notice/alert.json",
    "info": "https://www-bousai1.kenbousai-cloud.pref.aichi.jp/data/notice/info.json",
}
JR_TOKAI_URLS = {
    "operation": "https://traininfo.jr-central.co.jp/zairaisen/data/trainInfo/json/unkou.json",
    "notice": "https://traininfo.jr-central.co.jp/zairaisen/data/notice/json/oshirase.json",
}
IHIGHWAY_URLS = {
    "traffic": "https://www.c-ihighway.jp/datas/json/traffic.json",
    "traffic_count": "https://www.c-ihighway.jp/datas/json/trafficCount.json",
    "updated": "https://www.c-ihighway.jp/datas/json/updated.json",
    "important": "https://www.c-ihighway.jp/datas/json/importantInfo.json",
}
PUBLIC_SOURCE_URLS = {
    "aichi_bousai": "https://www-bousai1.kenbousai-cloud.pref.aichi.jp/pub_web/portal-top/",
    "kotsu_city": "https://www.kotsu.city.nagoya.jp/rp/emergency/",
    "jr_tokai": "https://traininfo.jr-central.co.jp/zairaisen/",
    "ihighway": "https://www.c-ihighway.jp/pcsite/map?area=area05",
    "jma_aichi": "https://www.jma.go.jp/bosai/#pattern=default&area_type=offices&area_code=230000",
    "linimo": "https://www.linimo.jp//delay/",
    "aikan": "https://www.aikanrailway.co.jp/train/",
}
KOTSU_CITY_URLS = {
    "latest_traffic": "https://www.kotsu.city.nagoya.jp/datas/latest_traffic.json",
}


@dataclass(frozen=True)
class PageSource:
    source: str
    area: str
    category: str
    url: str
    note: str = ""
    public_url: str = ""


PAGE_SOURCES: tuple[PageSource, ...] = (
    PageSource(
        source="名古屋鉄道 運行情報",
        area="愛知県",
        category="railway",
        url="https://top.meitetsu.co.jp/em/?mediacd=012",
        note="名鉄の公式運行情報",
    ),
    PageSource(
        source=LINIMO_SOURCE_NAME,
        area="愛知県・長久手市周辺",
        category="railway",
        url=PUBLIC_SOURCE_URLS["linimo"],
        note="リニモの公式運行情報",
    ),
    PageSource(
        source=AIKAN_SOURCE_NAME,
        area="愛知県・岡崎市から春日井市周辺",
        category="railway",
        url="https://www.aikanrailway.co.jp/AikanJsp/ZaisenInfo3.jsp",
        note="愛知環状鉄道の公式列車運行情報",
        public_url=PUBLIC_SOURCE_URLS["aikan"],
    ),
)


OFFICIAL_SOURCE_CATALOG = [
    {
        "key": "weather",
        "name": "気象庁・自治体防災情報",
        "area": "愛知県",
        "signal": "大雨・河川・浸水・避難情報に関する注意信号",
        "status": "live_feed_ready",
    },
    {
        "key": "municipal_disaster",
        "name": "愛知県・名古屋市 防災情報",
        "area": "愛知県・名古屋市",
        "signal": "自治体防災ポータルの災害・避難・緊急情報を低頻度で確認",
        "status": "official_page_poll",
    },
    {
        "key": "railway",
        "name": "鉄道運行情報",
        "area": "愛知県内の主要路線",
        "signal": "JR東海・名鉄・名古屋市交通局・リニモ・愛知環状鉄道の公式情報を低頻度で確認",
        "status": "official_page_or_json_poll",
    },
    {
        "key": "road",
        "name": "道路交通情報",
        "area": "愛知県・東海地方の高速道路",
        "signal": "NEXCO中日本 iHighway の公式JSONから東海エリアの事故・渋滞・規制情報を確認",
        "status": "official_json_poll",
    },
]


OFFICIAL_STATUS_CATALOG = [
    {"status": "通常", "description": "平常運行・発表なし・解除など、現在の注意対象がない状態。"},
    {"status": "情報", "description": "公式情報の入口や告知など、個別の危険・障害を示さない状態。"},
    {"status": "注意", "description": "注意報や注意喚起など、利用者が状況確認すべき状態。"},
    {"status": "警戒", "description": "警報・特別警報・災害危険など、強い警戒が必要な状態。"},
    {"status": "運休", "description": "鉄道などの運転見合わせ・運休。"},
    {"status": "支障", "description": "道路規制・渋滞・遅延など、移動に支障がある状態。"},
    {"status": "取得不可", "description": "取得失敗または判読不能。"},
]

JMA_STATUS_KEYWORDS = [
    ("解除", "通常"),
    ("特別警報", "警戒"),
    ("警報", "警戒"),
    ("土砂災害", "警戒"),
    ("記録的短時間大雨", "警戒"),
    ("竜巻", "注意"),
    ("注意報", "注意"),
    ("大雨", "注意"),
    ("洪水", "注意"),
]


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def source_catalog() -> list[dict]:
    """Return source groups shown by the API."""
    return OFFICIAL_SOURCE_CATALOG


def official_source_names() -> list[str]:
    """Return concrete source names stored in the Evidence DB."""
    return [
        JMA_SOURCE_NAME,
        AICHI_BOUSAI_SOURCE_NAME,
        *[source.source for source in PAGE_SOURCES],
        KOTSU_CITY_SOURCE_NAME,
        JR_TOKAI_SOURCE_NAME,
        IHIGHWAY_SOURCE_NAME,
    ]


def official_status_catalog() -> list[dict]:
    """Return the official signal statuses used by the app."""
    return OFFICIAL_STATUS_CATALOG


def trim_raw_text(text: str, max_chars: int = MAX_OFFICIAL_RAW_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def html_to_text(html: str) -> str:
    without_scripts = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    without_tags = re.sub(r"(?s)<[^>]+>", " ", without_scripts)
    return re.sub(r"\s+", " ", unescape(without_tags)).strip()


def focus_page_text(source: PageSource, text: str) -> str:
    """Keep the operational section first when official pages contain long navigation."""
    anchors = {
        "名古屋市交通局 運行情報": ("現在の情報", "運行情報ページ"),
        "名古屋鉄道 運行情報": ("15分以上", "列車運行情報"),
        LINIMO_SOURCE_NAME: ("最終更新", "現在、"),
        AIKAN_SOURCE_NAME: ("現在", "ただいま列車"),
    }.get(source.source, ())
    for anchor in anchors:
        index = text.find(anchor)
        if index >= 0:
            start = max(0, index - 120)
            return text[start:start + MAX_OFFICIAL_RAW_CHARS]
    return text


def strip_bom(value: str) -> str:
    return value.lstrip("\ufeff")


async def fetch_json(client: httpx.AsyncClient, url: str) -> dict:
    resp = await client.get(url)
    resp.raise_for_status()
    return json.loads(strip_bom(resp.text))


def localized_value(items, key: str = "name", lang: str = "ja") -> str:
    if isinstance(items, str):
        return items
    if not isinstance(items, list):
        return ""
    for item in items:
        if isinstance(item, dict) and item.get("lang") == lang:
            return str(item.get(key) or "")
    for item in items:
        if isinstance(item, dict) and item.get(key):
            return str(item.get(key))
    return ""


def localized_message(items, lang: str = "ja") -> str:
    return localized_value(items, key="message", lang=lang)


def localized_text(items, lang: str = "ja") -> str:
    return localized_value(items, key="text", lang=lang)


def compact_json(data: dict, max_chars: int = 1200) -> str:
    return trim_raw_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), max_chars=max_chars)


def jma_status_from_text(text: str, fallback_title: str = "") -> str:
    """数値スコアを使わず、気象庁本文・見出しの語でstatusだけを決める。"""
    combined = f"{text}\n{fallback_title}"
    for keyword, status in JMA_STATUS_KEYWORDS:
        if keyword in combined:
            return status
    if "注意してください" in combined:
        return "注意"
    if fallback_title in {"気象特別警報・警報・注意報", "気象警報・注意報（Ｈ２７）"}:
        return "注意"
    return "情報"


def is_aichi_jma_entry(author_name: str) -> bool:
    return author_name == JMA_AICHI_AUTHOR


def jma_entry_link(entry: ET.Element, ns: dict[str, str]) -> str:
    link = entry.find("atom:link", ns)
    return link.get("href") if link is not None and link.get("href") else JMA_FEED_URL


def jma_entry_content(entry: ET.Element, ns: dict[str, str]) -> str:
    return entry.findtext("atom:content", default="", namespaces=ns).strip()


def jma_entry_author(entry: ET.Element, ns: dict[str, str]) -> str:
    return entry.findtext("atom:author/atom:name", default="", namespaces=ns).strip()


async def fetch_jma_feed_entries(client: httpx.AsyncClient) -> list[ET.Element]:
    resp = await client.get(JMA_FEED_URL)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    return root.findall("atom:entry", ns)


async def fetch_jma_aichi_signals(limit: int = 20) -> list[dict]:
    """Fetch Aichi-related JMA signals already shaped for the live weather API."""
    signals = []
    seen_messages = set()
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        for entry in await fetch_jma_feed_entries(client):
            title = entry.findtext("atom:title", default="", namespaces=ns)
            updated = entry.findtext("atom:updated", default=now_text(), namespaces=ns)
            author_name = jma_entry_author(entry, ns)
            content = jma_entry_content(entry, ns)
            href = jma_entry_link(entry, ns)
            if not is_aichi_jma_entry(author_name):
                continue
            message_key = (updated, content)
            if message_key in seen_messages:
                continue
            seen_messages.add(message_key)

            signals.append({
                "source": JMA_SOURCE_NAME,
                "source_url": PUBLIC_SOURCE_URLS["jma_aichi"],
                "area": JMA_AICHI_AREA,
                "label": title,
                "severity": 0.0,
                "status": jma_status_from_text(content, title),
                "detail": f"{content} / 発表機関: {author_name} / data: {href}",
                "observed_at": updated,
            })
            if len(signals) >= limit:
                break
    return signals


async def fetch_jma_aichi_records(limit: int = 20) -> list[dict]:
    """Fetch Aichi-related JMA records as raw material for LLM normalization."""
    records = []
    seen_messages = set()
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        for entry in await fetch_jma_feed_entries(client):
            title = entry.findtext("atom:title", default="", namespaces=ns)
            updated = entry.findtext("atom:updated", default=now_text(), namespaces=ns)
            author_name = jma_entry_author(entry, ns)
            content = jma_entry_content(entry, ns)
            href = jma_entry_link(entry, ns)
            if not is_aichi_jma_entry(author_name):
                continue
            message_key = (updated, content)
            if message_key in seen_messages:
                continue
            seen_messages.add(message_key)

            records.append({
                "source": JMA_SOURCE_NAME,
                "area": JMA_AICHI_AREA,
                "category": "weather",
                "url": href,
                "source_url": PUBLIC_SOURCE_URLS["jma_aichi"],
                "title": title,
                "raw_text": trim_raw_text(
                    "\n".join(part for part in [
                        f"発表機関: {author_name}",
                        f"標題: {title}",
                        f"本文: {content}",
                    ] if part)
                ),
                "observed_at": updated,
            })
            if len(records) >= limit:
                break
    return records


def summarize_aichi_weather_warn(data: dict) -> str:
    current = data.get("currentStatus") or {}
    active_areas = []
    for area in current.get("areaList") or []:
        level = str(area.get("maxLevel") or "0")
        details = area.get("detail") or []
        if level != "0" or details:
            detail_text = ", ".join(
                str(item.get("name") or item.get("kindName") or item)
                for item in details[:6]
            )
            active_areas.append(f"{area.get('areaName')} maxLevel={level} {detail_text}".strip())

    lines = [
        f"気象警報・注意報 update={data.get('update')} currentStatus.dateTime={current.get('dateTime')} alarmFlag={data.get('alarmFlag')}",
    ]
    if active_areas:
        lines.append("発表中の地域: " + " / ".join(active_areas[:30]))
    else:
        lines.append("発表中の警報・注意報は見つかりません。全市町村 maxLevel=0。")
    return "\n".join(lines)


def summarize_aichi_simple_status(name: str, data: dict) -> str:
    current = data.get("currentStatus") or {}
    areas = current.get("areaList") or []
    lines = [
        f"{name} update={data.get('update')} currentStatus.dateTime={current.get('dateTime')} alarmFlag={data.get('alarmFlag')}",
    ]
    if areas:
        lines.append("対象地域: " + " / ".join(compact_json(area, max_chars=180) for area in areas[:20]))
    else:
        lines.append(f"現在、{name}の対象地域は見つかりません。")
    return "\n".join(lines)


def summarize_aichi_disaster(data: dict) -> str:
    disasters = data.get("disasterList") or []
    lines = [
        f"ポータル掲載災害一覧 update={data.get('update')} disasterCount={data.get('disasterCount')}",
        "注意: この一覧は履歴・掲載中情報を含む可能性があるため、現在発生中の災害とは断定しない。",
    ]
    if not disasters:
        lines.append("災害一覧に表示対象はありません。")
        return "\n".join(lines)
    for item in disasters[:12]:
        lines.append(
            f"- {item.get('disasterName')} rawStatus={item.get('disasterStatus')} start={item.get('startDateTime')} end={item.get('endDateTime')} code={item.get('disasterCode')}"
        )
    return "\n".join(lines)


def summarize_aichi_notice(name: str, data: dict) -> str:
    payload = data.get(name) or {}
    notices = payload.get("noticeList") or []
    lines = [f"{name} notice update={payload.get('update')} expires={payload.get('expires')}"]
    if not notices:
        lines.append("現在表示する通知はありません。")
        return "\n".join(lines)
    for item in notices[:10]:
        lines.append(f"- {compact_json(item, max_chars=220)}")
    return "\n".join(lines)


def aichi_update_time(*payloads: dict) -> str:
    """Return the first explicit update time from Aichi disaster JSON payloads."""
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        update = payload.get("update")
        if update:
            return str(update)
        for value in payload.values():
            if isinstance(value, dict) and value.get("update"):
                return str(value["update"])
    return now_text()


async def fetch_aichi_bousai_records() -> list[dict]:
    """Fetch Aichi disaster portal JSON and summarize it for LLM normalization."""
    observed_at = now_text()
    raw_parts = []
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT}) as client:
        try:
            weather_warn, quake, tsunami, disaster, alert, info = await asyncio.gather(
                fetch_json(client, AICHI_BOUSAI_URLS["weather_warn"]),
                fetch_json(client, AICHI_BOUSAI_URLS["quake"]),
                fetch_json(client, AICHI_BOUSAI_URLS["tsunami"]),
                fetch_json(client, AICHI_BOUSAI_URLS["disaster"]),
                fetch_json(client, AICHI_BOUSAI_URLS["alert"]),
                fetch_json(client, AICHI_BOUSAI_URLS["info"]),
            )
            raw_parts.append(summarize_aichi_weather_warn(weather_warn))
            raw_parts.append(summarize_aichi_simple_status("地震", quake))
            raw_parts.append(summarize_aichi_simple_status("津波", tsunami))
            raw_parts.append(summarize_aichi_disaster(disaster))
            raw_parts.append(summarize_aichi_notice("alert", alert))
            raw_parts.append(summarize_aichi_notice("info", info))
            title = "愛知県防災Web JSON"
            raw_text = "\n\n".join(raw_parts)
            observed_at = aichi_update_time(weather_warn, quake, tsunami, disaster, alert, info)
        except Exception as exc:
            title = f"{AICHI_BOUSAI_SOURCE_NAME} 取得失敗"
            raw_text = f"{type(exc).__name__}: {exc}"
    return [{
        "source": AICHI_BOUSAI_SOURCE_NAME,
        "area": "愛知県",
        "category": "disaster",
        "url": AICHI_BOUSAI_URLS["weather_warn"],
        "source_url": PUBLIC_SOURCE_URLS["aichi_bousai"],
        "title": title,
        "raw_text": trim_raw_text(raw_text),
        "observed_at": observed_at,
    }]


def summarize_jr_tokai_operation(data: dict) -> str:
    lines = [
        f"JR東海 在来線運行情報 create_time={jr_tokai_create_time_label(data.get('create_time'))} ono={data.get('ono')} check={data.get('check')}",
    ]
    messages = []
    for item in data.get("message_info") or []:
        line_name = localized_value(item.get("trainline"))
        message = localized_message(item.get("delivery_msg"))
        if message:
            messages.append(f"{line_name}: {message}")
    if messages:
        lines.append("公式メッセージ:")
        lines.extend(f"- {message}" for message in messages[:20])

    events = data.get("events") or []
    if events:
        lines.append("運行支障イベント:")
    for event in events[:30]:
        line_name = localized_value(event.get("imp_line"))
        status = localized_value(event.get("status"))
        cause = localized_value(event.get("cause"))
        section_from = localized_value(event.get("imp_sec_from"))
        section_to = localized_value(event.get("imp_sec_to"))
        direction = localized_value(event.get("direction"))
        prospect = localized_text(event.get("prospect_txt"))
        resume = event.get("resume_time") or event.get("resume_txt") or ""
        section = f"{section_from}～{section_to}" if section_from or section_to else ""
        parts = [
            f"{line_name}",
            f"status={status}",
            f"section={section}",
            f"direction={direction}",
            f"cause={cause}",
            f"prospect={prospect}",
            f"resume={resume}",
        ]
        lines.append("- " + " / ".join(part for part in parts if part and not part.endswith("=")))
    if not messages and not events:
        lines.append("現在、運行支障の公式メッセージは見つかりません。")
    return "\n".join(lines)


def summarize_kotsu_city_latest_traffic(data: list[dict]) -> str:
    """Summarize Nagoya Transportation Bureau's dynamic traffic JSON."""
    route_names = {
        "H_LINE": "東山線",
        "M_LINE": "名城線",
        "MK_LINE": "名港線",
        "T_LINE": "鶴舞線",
        "S_LINE": "桜通線",
        "K_LINE": "上飯田線",
        "B_LINE": "市バス",
    }
    lines = [f"名古屋市交通局 最新運行情報 取得時刻={now_text()}"]
    for item in sorted(data, key=lambda value: value.get("slot_no") or 0):
        route_name = route_names.get(item.get("rosen_id"), item.get("rosen_id") or "不明路線")
        title = item.get("traffic_title") or "情報なし"
        message = item.get("traffic_message") or ""
        section = item.get("traffic_section") or ""
        cause = item.get("traffic_cause") or ""
        parts = [route_name, title, message, section, cause]
        lines.append("- " + " / ".join(str(part) for part in parts if part))
    return "\n".join(lines)


async def fetch_kotsu_city_records() -> list[dict]:
    """Fetch Nagoya Transportation Bureau dynamic JSON instead of stale HTML fallback."""
    observed_at = now_text()
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT}) as client:
        try:
            latest_traffic = await fetch_json(client, KOTSU_CITY_URLS["latest_traffic"])
            title = "名古屋市交通局 最新運行情報 JSON"
            raw_text = summarize_kotsu_city_latest_traffic(latest_traffic if isinstance(latest_traffic, list) else [])
        except Exception as exc:
            title = f"{KOTSU_CITY_SOURCE_NAME} 取得失敗"
            raw_text = f"{type(exc).__name__}: {exc}"
    return [{
        "source": KOTSU_CITY_SOURCE_NAME,
        "area": "名古屋市",
        "category": "railway",
        "url": KOTSU_CITY_URLS["latest_traffic"],
        "source_url": PUBLIC_SOURCE_URLS["kotsu_city"],
        "title": title,
        "raw_text": trim_raw_text(raw_text),
        "observed_at": observed_at,
    }]


def summarize_jr_tokai_notice(data: dict) -> str:
    notices = []
    for item in data.get("notice_info") or []:
        line_name = localized_value(item.get("trainline"))
        for notice in item.get("notice_infoline") or []:
            message = next(
                (
                    entry
                    for entry in notice.get("notice_message") or []
                    if isinstance(entry, dict) and entry.get("lang") == "ja"
                ),
                {},
            )
            title = message.get("title") or ""
            if title:
                notices.append(
                    f"{line_name}: {title} publication={notice.get('publication_from')}～{notice.get('publication_to')}"
                )
    if not notices:
        return "JR東海 お知らせ: 表示対象なし"
    return "JR東海 お知らせ:\n" + "\n".join(f"- {notice}" for notice in notices[:20])


def jr_tokai_create_time_with_clock(value) -> str:
    """JR東海 create_time は日付のみの場合があるため、時刻つきの値だけ採用する。"""
    raw = str(value or "").strip()
    if not raw:
        return ""
    if re.fullmatch(r"\d{8}", raw):
        return ""
    if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", raw):
        return ""
    return raw


def jr_tokai_create_time_label(value) -> str:
    """LLMに日付のみを正確な時刻として解釈させないための表示用文字列。"""
    raw = str(value or "").strip()
    if not raw:
        return "時刻未提供"
    if not jr_tokai_create_time_with_clock(raw):
        return f"{raw}（日付のみ・正確な更新時刻なし）"
    return raw


async def fetch_jr_tokai_records() -> list[dict]:
    """Fetch JR Central JSON and summarize it for LLM normalization."""
    observed_at = now_text()
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT}) as client:
        try:
            operation, notice = await asyncio.gather(
                fetch_json(client, JR_TOKAI_URLS["operation"]),
                fetch_json(client, JR_TOKAI_URLS["notice"]),
            )
            title = "JR東海 在来線運行情報 JSON"
            raw_text = summarize_jr_tokai_operation(operation) + "\n\n" + summarize_jr_tokai_notice(notice)
            observed_at = jr_tokai_create_time_with_clock(operation.get("create_time")) or observed_at
        except Exception as exc:
            title = f"{JR_TOKAI_SOURCE_NAME} 取得失敗"
            raw_text = f"{type(exc).__name__}: {exc}"
    return [{
        "source": JR_TOKAI_SOURCE_NAME,
        "area": "愛知県・東海地方",
        "category": "railway",
        "url": JR_TOKAI_URLS["operation"],
        "source_url": PUBLIC_SOURCE_URLS["jr_tokai"],
        "title": title,
        "raw_text": trim_raw_text(raw_text),
        "observed_at": observed_at,
    }]


def summarize_ihighway_traffic(area: dict, count_info: dict, updated: dict, important: dict) -> str:
    lines = [
        f"iHighway 東海エリア updated={updated.get('updated')} data={updated.get('data')} setting={updated.get('setting')}",
        f"trafficCount summary={compact_json(count_info, max_chars=500)}",
    ]
    important_items = important.get("info") or []
    if important_items:
        lines.append("重要情報:")
        lines.extend(f"- {compact_json(item, max_chars=260)}" for item in important_items[:10])

    traffic_info = area.get("trafficInfo") or {}
    labels = {
        "closed": "通行止",
        "ramp": "入口出口規制",
        "snowChain": "チェーン規制",
        "snowTires": "冬用タイヤ規制",
        "jam": "渋滞",
        "oneLane": "片側交互通行",
        "accident": "事故",
    }
    for traffic_kind, roads in traffic_info.items():
        label = labels.get(traffic_kind, traffic_kind)
        for road in roads or []:
            road_name = road.get("roadName") or ""
            for info in road.get("info") or road.get("ic") or []:
                if "info" in info and isinstance(info.get("info"), list):
                    nested_items = info.get("info")
                else:
                    nested_items = [info]
                for item in nested_items:
                    parts = [
                        label,
                        road_name,
                        item.get("title"),
                        item.get("direction"),
                        item.get("reason"),
                        f"{item.get('distance')}km" if item.get("distance") else "",
                    ]
                    lines.append("- " + " / ".join(str(part) for part in parts if part))

    other_info = area.get("otherTrafficInfo") or {}
    other_labels = {
        "laneRestriction": "車線規制",
        "underRegulation": "規制中",
        "speed": "速度規制",
    }
    for traffic_kind, roads in other_info.items():
        label = other_labels.get(traffic_kind, traffic_kind)
        for road in roads or []:
            road_name = road.get("roadName") or ""
            for item in road.get("info") or []:
                parts = [
                    label,
                    road_name,
                    item.get("title"),
                    item.get("direction"),
                    item.get("reason"),
                    item.get("detail"),
                ]
                lines.append("- " + " / ".join(str(part) for part in parts if part))

    if len(lines) == 2:
        lines.append("東海エリアの交通障害データは見つかりません。")
    return "\n".join(lines[:120])


async def fetch_ihighway_tokai_records() -> list[dict]:
    """Fetch iHighway JSON for area05 Tokai and summarize it for LLM normalization."""
    observed_at = now_text()
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT}) as client:
        try:
            traffic, traffic_count, updated, important = await asyncio.gather(
                fetch_json(client, IHIGHWAY_URLS["traffic"]),
                fetch_json(client, IHIGHWAY_URLS["traffic_count"]),
                fetch_json(client, IHIGHWAY_URLS["updated"]),
                fetch_json(client, IHIGHWAY_URLS["important"]),
            )
            title = "iHighway 中日本 東海エリア JSON"
            raw_text = summarize_ihighway_traffic(
                traffic.get("area05") or {},
                traffic_count.get("area05") or {},
                updated,
                important,
            )
            observed_at = str(updated.get("updated") or observed_at)
        except Exception as exc:
            title = f"{IHIGHWAY_SOURCE_NAME} 取得失敗"
            raw_text = f"{type(exc).__name__}: {exc}"
    return [{
        "source": IHIGHWAY_SOURCE_NAME,
        "area": "東海地方・高速道路",
        "category": "expressway",
        "url": IHIGHWAY_URLS["traffic"],
        "source_url": PUBLIC_SOURCE_URLS["ihighway"],
        "title": title,
        "raw_text": trim_raw_text(raw_text),
        "observed_at": observed_at,
    }]


async def fetch_page_record(client: httpx.AsyncClient, source: PageSource) -> dict:
    """Fetch one official web page as raw text for LLM normalization."""
    observed_at = now_text()
    try:
        resp = await client.get(source.url)
        resp.raise_for_status()
        title = source.source
        raw_text = trim_raw_text(focus_page_text(source, html_to_text(resp.text)))
    except Exception as exc:
        title = f"{source.source} 取得失敗"
        raw_text = f"{type(exc).__name__}: {exc}"
    return {
        "source": source.source,
        "area": source.area,
        "category": source.category,
        "url": source.url,
        "source_url": source.public_url or source.url,
        "title": title,
        "raw_text": raw_text,
        "observed_at": observed_at,
    }


async def fetch_official_page_records() -> list[dict]:
    """Fetch raw text from official pages that do not expose a simple public API."""
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(
        timeout=HTTP_TIMEOUT_SECONDS,
        follow_redirects=True,
        headers=headers,
    ) as client:
        return [await fetch_page_record(client, source) for source in PAGE_SOURCES]


async def fetch_raw_official_records(limit: int = 20) -> list[dict]:
    """Fetch all raw official records used by the Evidence DB sync pipeline."""
    records = []
    records.extend(await fetch_jma_aichi_records(limit=limit))
    records.extend(await fetch_aichi_bousai_records())
    records.extend(await fetch_official_page_records())
    records.extend(await fetch_kotsu_city_records())
    records.extend(await fetch_jr_tokai_records())
    records.extend(await fetch_ihighway_tokai_records())
    return records


def official_summary(signals: list[dict]) -> dict:
    """Summarize normalized official signals."""
    if not signals:
        return {"sources": [], "status": "通常"}

    sources = sorted({signal["source"] for signal in signals})
    statuses = {signal.get("status") for signal in signals}

    if "警戒" in statuses or "運休" in statuses:
        status = "警戒"
    elif "支障" in statuses:
        status = "支障"
    elif "注意" in statuses:
        status = "注意"
    elif statuses <= {"通常", "情報", None}:
        status = "通常"
    else:
        status = "情報"

    return {"sources": sources, "status": status}

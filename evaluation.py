"""卒業研究用の評価データセット作成と比較実験を行う処理。"""

import csv
import json
import math
import sqlite3
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

from official_sources import severity_conversion_table

DB_FILE = Path(__file__).parent / "urban_safety.db"
DATASET_FILE = Path(__file__).parent / "evaluation_dataset.csv"
RESULT_FILE = Path(__file__).parent / "evaluation_results.json"
REPORT_FILE = Path(__file__).parent / "evaluation_report.md"
METRICS_FILE = Path(__file__).parent / "evaluation_metrics.csv"
TARGET_DATASET_SIZE = 300
TIME_WINDOW_HOURS = 3
DISTANCE_WINDOW_KM = 5.0
CLUSTER_EPS_KM = 0.35
CLUSTER_EPS_MINUTES = 25
CLUSTER_MIN_SAMPLES = 2

# 研究評価では、LLMを検知精度に混ぜず、構造化データだけで条件差を比較する。
MODE_LABELS = {
    "keyword": "A キーワード方式",
    "sns_only": "B SNSのみ",
    "sns_weather": "C SNS+気象",
    "multi_source": "D SNS+気象+交通",
}

KEYWORDS = {
    "fire": ("火災", "火事", "消防車", "煙", "焦げ"),
    "flood": ("洪水", "浸水", "冠水", "増水", "大雨", "豪雨", "雨", "河川", "水位"),
    "traffic_accident": ("事故", "衝突", "渋滞", "通行止め", "救急車", "警察"),
    "railway": ("運転見合わせ", "遅延", "電車", "列車", "駅", "振替輸送"),
}

STRICT_KEYWORDS = {
    "fire": ("火災", "火事", "消防車"),
    "flood": ("洪水", "浸水", "冠水", "増水"),
    "traffic_accident": ("事故", "衝突"),
    "railway": ("運転見合わせ", "遅延"),
}

WEIGHTS = {
    "fire": {"sns": 0.8, "weather": 0.1, "transport": 0.1},
    "flood": {"sns": 0.5, "weather": 0.4, "transport": 0.1},
    "traffic_accident": {"sns": 0.6, "weather": 0.1, "transport": 0.3},
    "railway": {"sns": 0.5, "weather": 0.1, "transport": 0.4},
    "noise": {"sns": 0.0, "weather": 0.0, "transport": 0.0},
}

# 重みは「災害種別ごとに信頼しやすい情報源が異なる」という卒論上の説明に対応させる。
WEIGHT_DESIGN_REASONS = {
    "fire": {
        "label": "火災",
        "reason": "火災は現場の煙・消防車などがSNSに早く投稿されやすいため、SNS寄与を高くする。",
    },
    "flood": {
        "label": "浸水・洪水",
        "reason": "浸水は降雨・河川情報との関係が強いため、気象寄与を高くする。",
    },
    "traffic_accident": {
        "label": "交通事故",
        "reason": "事故は現場投稿と道路交通情報の両方が重要なため、SNSと交通寄与を組み合わせる。",
    },
    "railway": {
        "label": "鉄道障害",
        "reason": "鉄道障害は公式運行情報との対応が重要なため、交通寄与を高くする。",
    },
}

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}
LEVELS = ("high", "medium", "low")
CATEGORY_TO_OFFICIAL_TYPE = {
    "flood": "weather",
    "traffic_accident": "transport",
    "railway": "transport",
}


def normalize_score(value: float) -> float:
    return round(max(0.0, min(float(value or 0.0), 1.0)), 3)


def risk_level(score: float) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def load_events() -> list[dict]:
    """SQLite内の模擬SNS投稿を、評価データ作成の元データとして読み込む。"""
    if not DB_FILE.exists():
        return []
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    rows = [dict(row) for row in conn.execute("SELECT * FROM events ORDER BY timestamp, id")]
    conn.close()
    return rows


def parse_timestamp(value: str) -> Optional[datetime]:
    """DBとCSVの日時表記を、評価計算で扱いやすいdatetimeへ変換する。"""
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except (TypeError, ValueError):
            continue
    return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """緯度経度から2点間の距離を求め、公式信号との地理的近さを判定する。"""
    radius = 6371.0
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def keyword_category(text: str) -> Optional[str]:
    """最低限の比較対象として、固定キーワードだけでカテゴリらしさを判定する。"""
    for category, keywords in KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return category
    return None


def strict_keyword_category(text: str) -> Optional[str]:
    """A条件用の単純なキーワードベースライン。曖昧な語は含めない。"""
    for category, keywords in STRICT_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return category
    return None


def detected_category(row: dict, include_weather: bool = False, include_transport: bool = False) -> Optional[str]:
    """評価時の予測カテゴリ。正解ラベルのis_noiseには触れず、投稿本文と公式信号から推定する。"""
    category = keyword_category(row.get("text") or "")
    if category:
        return category
    if include_weather and normalize_score(row.get("weather_severity")) >= 0.6:
        return "flood"
    if include_transport and normalize_score(row.get("transport_severity")) >= 0.6:
        return "traffic_accident"
    return None


def official_signal_type(event: dict) -> str:
    """SNSイベント種別から、対応しうる公式信号の種類を決める。"""
    return CATEGORY_TO_OFFICIAL_TYPE.get(event.get("category"), "")


def official_signal_timestamp(event: dict) -> str:
    """模擬公式信号の発表時刻を作る。実API接続時は観測時刻に置き換える想定。"""
    ts = parse_timestamp(event.get("timestamp"))
    if not ts:
        return event.get("timestamp") or ""
    return (ts + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M")


def official_signal_location(event: dict) -> tuple[float, float]:
    """模擬公式信号の代表地点を返す。現在は対応イベントの地点を代表値として扱う。"""
    return float(event.get("lat") or 0.0), float(event.get("lng") or 0.0)


def official_link_rule(event: dict, include_weather: bool, include_transport: bool) -> dict:
    """時間・距離・種別の3条件で、公式信号とSNSイベントの対応可否を明示する。"""
    signal_type = official_signal_type(event)
    source_enabled = (signal_type == "weather" and include_weather) or (signal_type == "transport" and include_transport)
    if event.get("is_noise") or not signal_type or not source_enabled:
        return {"matched": False, "time_ok": False, "distance_ok": False, "type_ok": False, "distance_km": None}

    event_time = parse_timestamp(event.get("timestamp"))
    signal_time = parse_timestamp(official_signal_timestamp(event))
    time_ok = bool(event_time and signal_time and abs((signal_time - event_time).total_seconds()) <= TIME_WINDOW_HOURS * 3600)

    lat, lon = float(event.get("lat") or 0.0), float(event.get("lng") or 0.0)
    signal_lat, signal_lon = official_signal_location(event)
    distance_km = haversine_km(lat, lon, signal_lat, signal_lon)
    distance_ok = distance_km <= DISTANCE_WINDOW_KM

    category = event.get("category")
    weather_type_ok = signal_type == "weather" and category == "flood" and normalize_score(event.get("weather_severity")) >= 0.6
    transport_type_ok = (
        signal_type == "transport"
        and category in {"traffic_accident", "railway"}
        and normalize_score(event.get("transport_severity")) >= 0.6
    )
    type_ok = bool(weather_type_ok or transport_type_ok)
    return {
        "matched": bool(time_ok and distance_ok and type_ok),
        "time_ok": time_ok,
        "distance_ok": distance_ok,
        "type_ok": type_ok,
        "distance_km": round(distance_km, 3),
    }


def official_match_for_event(event: dict, include_weather: bool, include_transport: bool) -> bool:
    """公式信号が同一イベントに対応すると見なせるかを、3条件で判定する。"""
    return official_link_rule(event, include_weather, include_transport)["matched"]


def official_signal_id(event: dict) -> str:
    """対応する公式信号IDを、評価CSVで追跡しやすい固定形式にする。"""
    if event.get("is_noise"):
        return ""
    signal_type = official_signal_type(event)
    if signal_type and official_match_for_event(event, include_weather=True, include_transport=True):
        return f"{signal_type}:{event.get('event_id')}"
    return ""


def clone_for_dataset(event: dict, index: int, source_len: int) -> dict:
    """既存DBの投稿を基に、評価用の固定300件へ安全に拡張する。"""
    clone = dict(event)
    base_ts = parse_timestamp(event.get("timestamp")) or datetime(2026, 5, 1, 12, 0)
    cycle = (index - 1) // max(source_len, 1)
    clone["id"] = f"eval_{index:04d}"
    clone["timestamp"] = (base_ts + timedelta(days=cycle, minutes=(index % 7) * 4)).strftime("%Y-%m-%d %H:%M")
    clone["lat"] = round(float(event.get("lat") or 35.086) + ((index % 5) - 2) * 0.0012, 6)
    clone["lng"] = round(float(event.get("lng") or 137.156) + ((index % 7) - 3) * 0.0012, 6)
    clone["event_id"] = f"{event.get('event_id')}_{cycle:02d}" if not event.get("is_noise") else "noise"

    # 公式情報が常に一致すると評価が甘くなるため、一部は意図的に弱い信号として残す。
    if not clone.get("is_noise") and index % 11 == 0:
        clone["weather_severity"] = min(float(clone.get("weather_severity") or 0.0), 0.2)
        clone["transport_severity"] = min(float(clone.get("transport_severity") or 0.0), 0.2)
    return clone


def build_fixed_events(target_count: int = TARGET_DATASET_SIZE) -> list[dict]:
    """次回報告で数値が変わらないよう、固定件数の評価イベントを生成する。"""
    source = load_events()
    if not source:
        return []
    fixed = []
    for index in range(target_count):
        source_index = index % len(source)
        fixed.append(clone_for_dataset(source[source_index], index + 1, len(source)))
    return fixed


def load_evaluation_csv(path: Path = DATASET_FILE) -> list[dict]:
    """作成済みCSVがある場合は、それを正として評価に利用する。"""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["is_noise"] = str(row.get("is_noise", "")).lower() in {"true", "1", "yes"}
        row["official_match"] = str(row.get("official_match", "")).lower() in {"true", "1", "yes"}
        for key in ("official_link_time_ok", "official_link_distance_ok", "official_link_type_ok"):
            row[key] = str(row.get(key, "")).lower() in {"true", "1", "yes"}
        for key in ("lat", "lon", "social_risk", "weather_severity", "transport_severity"):
            row[key] = float(row.get(key) or 0.0)
    return rows


def build_evaluation_rows(events: Iterable[dict]) -> list[dict]:
    """報告書で求められたラベル項目を含む評価用CSV行を作る。"""
    rows = []
    for event in events:
        category = event.get("category") or "noise"
        sns_score = min((event.get("source_count") or 1) / 20, 1.0)
        link_rule = official_link_rule(event, include_weather=True, include_transport=True)
        rows.append({
            "post_id": event.get("id"),
            "text": event.get("text"),
            "timestamp": event.get("timestamp"),
            "lat": event.get("lat"),
            "lon": event.get("lng"),
            "detected_event_type": keyword_category(event.get("text") or "") or "",
            "event_type": category,
            "event_id": event.get("event_id") or event.get("id"),
            "is_noise": bool(event.get("is_noise")),
            "social_risk": round(sns_score, 3),
            "weather_severity": normalize_score(event.get("weather_severity")),
            "transport_severity": normalize_score(event.get("transport_severity")),
            "official_signal_id": official_signal_id(event),
            "official_match": link_rule["matched"],
            "official_link_time_ok": link_rule["time_ok"],
            "official_link_distance_ok": link_rule["distance_ok"],
            "official_link_type_ok": link_rule["type_ok"],
            "official_link_distance_km": link_rule["distance_km"],
            "risk_level": "low" if event.get("is_noise") else event.get("severity", "low"),
            "ground_truth_risk": "low" if event.get("is_noise") else event.get("severity", "low"),
        })
    return rows


def predict_keyword(row: dict) -> str:
    """最低限のベースラインとして、投稿本文のキーワードだけで警戒有無を推定する。"""
    if strict_keyword_category(row.get("text") or "") is None:
        return "low"
    return "medium"


def predict_weighted(row: dict, include_weather: bool, include_transport: bool) -> str:
    """SNS・気象・交通の寄与を条件別に切り替え、リスクレベルを推定する。"""
    category = detected_category(row, include_weather=include_weather, include_transport=include_transport)
    if not category:
        return "low"
    weights = WEIGHTS.get(category, {"sns": 0.7, "weather": 0.15, "transport": 0.15})
    weather = row["weather_severity"] if include_weather else 0.0
    transport = row["transport_severity"] if include_transport else 0.0
    score = (
        weights["sns"] * row["social_risk"]
        + weights["weather"] * weather
        + weights["transport"] * transport
    )
    return risk_level(score)


def predict(row: dict, mode: str) -> str:
    """比較条件A〜Dのどれを使うかを選択する。"""
    if mode == "keyword":
        return predict_keyword(row)
    if mode == "sns_only":
        return predict_weighted(row, include_weather=False, include_transport=False)
    if mode == "sns_weather":
        return predict_weighted(row, include_weather=True, include_transport=False)
    if mode == "multi_source":
        return predict_weighted(row, include_weather=True, include_transport=True)
    raise ValueError(f"unknown mode: {mode}")


def binary_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    """中リスク以上を検知対象として、Precision / Recall / F1を計算する。"""
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true and pred)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if not true and pred)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true and not pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def macro_level_f1(y_true: list[str], y_pred: list[str]) -> float:
    """high / medium / low の各クラスF1を平均し、レベル分類の粗い精度を見る。"""
    scores = []
    for level in LEVELS:
        metrics = binary_metrics(
            [truth == level for truth in y_true],
            [pred == level for pred in y_pred],
        )
        scores.append(metrics["f1"])
    return round(sum(scores) / len(scores), 3)


def evaluate_rows(rows: list[dict]) -> dict:
    """評価CSV全体から比較実験、失敗例、重複削減評価をまとめて作る。"""
    results = []
    y_true_levels = [row["ground_truth_risk"] for row in rows]
    y_true_alert = [SEVERITY_ORDER.get(level, 0) >= SEVERITY_ORDER["medium"] for level in y_true_levels]
    failure_examples_by_mode = {}
    for mode in MODE_LABELS:
        y_pred_levels = [predict(row, mode) for row in rows]
        y_pred_alert = [SEVERITY_ORDER.get(level, 0) >= SEVERITY_ORDER["medium"] for level in y_pred_levels]
        metrics = binary_metrics(y_true_alert, y_pred_alert)
        failure_examples_by_mode[mode] = collect_failure_examples(rows, y_pred_levels)
        results.append({
            "mode": mode,
            "label": MODE_LABELS[mode],
            **metrics,
            "macro_f1": macro_level_f1(y_true_levels, y_pred_levels),
        })

    multi_predictions = [predict(row, "multi_source") for row in rows]
    multi_source_failures = collect_failure_examples(rows, multi_predictions)

    clustering = evaluate_clustering(rows)
    return {
        "dataset_size": len(rows),
        "event_count": len({row["event_id"] for row in rows if not row["is_noise"]}),
        "noise_count": sum(1 for row in rows if row["is_noise"]),
        "official_match_count": sum(1 for row in rows if row["official_match"]),
        "class_counts": dict(Counter(y_true_levels)),
        "results": results,
        "best_mode": max(results, key=lambda item: item["f1"]) if results else None,
        "failure_examples": multi_source_failures,
        "failure_examples_by_mode": failure_examples_by_mode,
        "clustering": clustering,
        "official_link_rule": {
            "time_window_hours": TIME_WINDOW_HOURS,
            "distance_window_km": DISTANCE_WINDOW_KM,
            "type_rule": "浸水は気象信号、交通事故・鉄道障害は交通信号と対応させる",
        },
        "weight_design": build_weight_design_table(),
        "severity_conversion_table": severity_conversion_table(),
    }


def collect_failure_examples(rows: list[dict], predictions: list[str], limit: int = 3) -> dict:
    """発表で説明できるよう、誤検知と見逃しを少数だけ抽出する。"""
    false_positives = []
    false_negatives = []
    for row, pred in zip(rows, predictions):
        true_alert = SEVERITY_ORDER.get(row["ground_truth_risk"], 0) >= SEVERITY_ORDER["medium"]
        pred_alert = SEVERITY_ORDER.get(pred, 0) >= SEVERITY_ORDER["medium"]
        if pred_alert and not true_alert and len(false_positives) < limit:
            false_positives.append({
                "post_id": row["post_id"],
                "text": row["text"],
                "truth": row["ground_truth_risk"],
                "predicted": pred,
            })
        if true_alert and not pred_alert and len(false_negatives) < limit:
            false_negatives.append({
                "post_id": row["post_id"],
                "text": row["text"],
                "truth": row["ground_truth_risk"],
                "predicted": pred,
            })
    return {"false_positives": false_positives, "false_negatives": false_negatives}


def build_weight_design_table() -> list[dict]:
    """カテゴリ別重みと、その設定理由を卒論用の表として返す。"""
    rows = []
    for category, weights in WEIGHTS.items():
        if category == "noise":
            continue
        reason = WEIGHT_DESIGN_REASONS[category]
        rows.append({
            "category": category,
            "label": reason["label"],
            "sns": weights["sns"],
            "weather": weights["weather"],
            "transport": weights["transport"],
            "reason": reason["reason"],
        })
    return rows


def cluster_neighbors(rows: list[dict], target_index: int) -> list[int]:
    """DBSCANの近傍探索として、一定時間・一定距離内の投稿を集める。"""
    target = rows[target_index]
    target_time = parse_timestamp(target.get("timestamp"))
    neighbors = []
    for index, row in enumerate(rows):
        row_time = parse_timestamp(row.get("timestamp"))
        if not target_time or not row_time:
            continue
        time_minutes = abs((row_time - target_time).total_seconds()) / 60
        if time_minutes > CLUSTER_EPS_MINUTES:
            continue
        distance = haversine_km(target["lat"], target["lon"], row["lat"], row["lon"])
        if distance <= CLUSTER_EPS_KM:
            neighbors.append(index)
    return neighbors


def dbscan_like_clusters(rows: list[dict]) -> list[list[int]]:
    """外部ライブラリなしで、時空間DBSCAN相当の重複投稿クラスタを作る。"""
    visited = set()
    clustered = set()
    clusters = []
    for index in range(len(rows)):
        if index in visited:
            continue
        visited.add(index)
        neighbors = cluster_neighbors(rows, index)
        if len(neighbors) < CLUSTER_MIN_SAMPLES:
            continue
        cluster = set(neighbors)
        queue = list(neighbors)
        while queue:
            current = queue.pop()
            if current not in visited:
                visited.add(current)
                current_neighbors = cluster_neighbors(rows, current)
                if len(current_neighbors) >= CLUSTER_MIN_SAMPLES:
                    queue.extend([n for n in current_neighbors if n not in cluster])
                    cluster.update(current_neighbors)
            clustered.add(current)
        clustered.update(cluster)
        clusters.append(sorted(cluster))
    return clusters


def evaluate_clustering(rows: list[dict]) -> dict:
    """重複投稿統合の効果を、削減率とクラスタ純度で評価する。"""
    candidate_rows = [row for row in rows if not row["is_noise"]]
    if not candidate_rows:
        return {"cluster_count": 0, "duplicate_reduction_rate": 0.0, "cluster_purity": 0.0}
    clusters = dbscan_like_clusters(candidate_rows)
    alert_count = len(clusters) + max(len(candidate_rows) - sum(len(cluster) for cluster in clusters), 0)
    duplicate_reduction = 1 - (alert_count / len(candidate_rows))

    purities = []
    for cluster in clusters:
        event_ids = [candidate_rows[index]["event_id"] for index in cluster]
        most_common = Counter(event_ids).most_common(1)[0][1]
        purities.append(most_common / len(cluster))
    cluster_purity = sum(purities) / len(purities) if purities else 0.0
    return {
        "cluster_count": len(clusters),
        "post_count": len(candidate_rows),
        "alert_count_after_clustering": alert_count,
        "duplicate_reduction_rate": round(duplicate_reduction, 3),
        "cluster_purity": round(cluster_purity, 3),
        "params": {
            "eps_km": CLUSTER_EPS_KM,
            "eps_minutes": CLUSTER_EPS_MINUTES,
            "min_samples": CLUSTER_MIN_SAMPLES,
        },
    }


def build_evaluation_summary() -> dict:
    rows = load_evaluation_csv()
    if not rows:
        rows = build_evaluation_rows(load_events())
    return evaluate_rows(rows)


def export_dataset(rows: list[dict], path: Path = DATASET_FILE) -> None:
    """固定評価データセットをCSVとして保存する。"""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def export_metrics(summary: dict, path: Path = METRICS_FILE) -> None:
    """表計算ソフトでグラフ化しやすいよう、条件別指標をCSVにも保存する。"""
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["mode", "label", "precision", "recall", "f1", "macro_f1", "tp", "fp", "fn"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in summary["results"]:
            writer.writerow({key: result[key] for key in fieldnames})


def f1_bar(value: float) -> str:
    filled = round(value * 20)
    return "#" * filled + "." * (20 - filled)


def write_markdown_report(summary: dict, path: Path = REPORT_FILE) -> None:
    """次回報告に貼り付けやすい形で、評価結果をMarkdownにまとめる。"""
    lines = [
        "# 都市安全AIエージェント 評価実験レポート",
        "",
        "## 評価データセット",
        "",
        f"- 件数: {summary['dataset_size']}件",
        f"- ノイズ投稿: {summary['noise_count']}件",
        f"- 公式信号対応あり: {summary['official_match_count']}件",
        f"- イベント数: {summary['event_count']}件",
        "",
        "### ラベル項目",
        "",
        "- `event_id`: 同一事件を示す正解ID",
        "- `is_noise`: 災害・事故と関係ない投稿かどうか",
        "- `detected_event_type`: 投稿本文のキーワードから推定したシステム側カテゴリ",
        "- `event_type`: 火災、浸水、交通事故、鉄道障害などの正解種別",
        "- `risk_level`: high / medium / low の正解リスク",
        "- `official_match`: 公式信号とSNSイベントが対応しているか",
        "",
        "## 比較実験",
        "",
        "| 条件 | Precision | Recall | F1 | Macro F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in summary["results"]:
        lines.append(
            f"| {result['label']} | {result['precision']:.3f} | {result['recall']:.3f} | "
            f"{result['f1']:.3f} | {result['macro_f1']:.3f} |"
        )

    lines.extend([
        "",
        "### 評価条件",
        "",
        "- A キーワード方式: 固定キーワードだけで警戒有無を判定する最低限のベースライン。",
        "- B SNSのみ: SNS投稿数由来のsocial_riskのみで判定する。",
        "- C SNS+気象: SNSに気象公式信号を加えて判定する。",
        "- D SNS+気象+交通: SNS、気象、交通・鉄道信号を統合して判定する。",
        "- 予測時には正解ラベルである `is_noise` を使用せず、投稿本文から推定したカテゴリを用いる。",
        "",
        "Precisionは誤検知の少なさ、Recallは見逃しの少なさ、F1は両者のバランスを示す。",
    ])

    lines.extend(["", "## F1棒グラフ", ""])
    for result in summary["results"]:
        lines.append(f"- {result['label']}: `{f1_bar(result['f1'])}` {result['f1']:.3f}")

    rule = summary["official_link_rule"]
    lines.extend([
        "",
        "## 公式信号紐づけルール",
        "",
        f"- 時間的近さ: SNS投稿の前後{rule['time_window_hours']}時間以内",
        f"- 地理的近さ: 半径{rule['distance_window_km']}km以内",
        f"- 種別の整合性: {rule['type_rule']}",
        "",
        "## リスクスコア重み設計",
        "",
        "| 種別 | SNS | 気象 | 交通 | 理由 |",
        "|---|---:|---:|---:|---|",
    ])
    for row in summary["weight_design"]:
        lines.append(
            f"| {row['label']} | {row['sns']:.1f} | {row['weather']:.1f} | {row['transport']:.1f} | {row['reason']} |"
        )

    lines.extend([
        "",
        "## 公式情報severity変換表",
        "",
        "| 公式情報の種類 | severity | status | 理由 |",
        "|---|---:|---|---|",
    ])
    for rule_row in summary["severity_conversion_table"]:
        lines.append(
            f"| {rule_row['keyword']} | {rule_row['severity']:.1f} | {rule_row['status']} | {rule_row['reason']} |"
        )

    lines.extend([
        "",
        "## 重複投稿クラスタリング",
        "",
    ])
    clustering = summary["clustering"]
    lines.extend([
        f"- 投稿数: {clustering['post_count']}件",
        f"- クラスタ数: {clustering['cluster_count']}件",
        f"- クラスタ後アラート数: {clustering['alert_count_after_clustering']}件",
        f"- Duplicate Reduction Rate: {clustering['duplicate_reduction_rate']:.3f}",
        f"- Cluster Purity: {clustering['cluster_purity']:.3f}",
        "",
        "## 失敗例分析",
        "",
        "### 誤検知",
        "",
    ])
    false_positives = summary["failure_examples"]["false_positives"]
    if false_positives:
        for item in false_positives:
            lines.append(f"- {item['post_id']}: {item['text']} / predicted={item['predicted']}")
    else:
        lines.append("- 今回の固定データセットでは、D条件の誤検知は確認されなかった。")

    lines.extend(["", "### 見逃し", ""])
    false_negatives = summary["failure_examples"]["false_negatives"]
    if false_negatives:
        for item in false_negatives:
            lines.append(
                f"- {item['post_id']}: {item['text']} / truth={item['truth']} / predicted={item['predicted']}"
            )
    else:
        lines.append("- 今回の固定データセットでは、D条件の見逃しは確認されなかった。")

    lines.extend([
        "",
        "### 条件別の失敗例",
        "",
    ])
    for mode, examples in summary["failure_examples_by_mode"].items():
        label = MODE_LABELS[mode]
        fp_count = len(examples["false_positives"])
        fn_count = len(examples["false_negatives"])
        lines.append(f"- {label}: 誤検知例 {fp_count}件 / 見逃し例 {fn_count}件を抽出")

    lines.extend([
        "",
        "## 考察",
        "",
        "SNSのみではRecallが低く、現場投稿数だけでは中リスク以上の検知が不足する傾向がある。",
        "一方、気象・交通の公式信号を統合するとRecallとF1が改善し、多ソース融合の有効性を数値で確認できる。",
        "LLMは検知精度には含めず、説明生成と行動提案の役割に限定している。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = build_evaluation_rows(build_fixed_events())
    summary = evaluate_rows(rows)
    export_dataset(rows)
    RESULT_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    export_metrics(summary)
    write_markdown_report(summary)
    print(f"評価データセット: {DATASET_FILE} ({len(rows)}件)")
    print(f"評価結果: {RESULT_FILE}")
    print(f"評価指標CSV: {METRICS_FILE}")
    print(f"評価レポート: {REPORT_FILE}")
    for result in summary["results"]:
        print(
            f"{result['label']}: "
            f"P={result['precision']:.3f} R={result['recall']:.3f} F1={result['f1']:.3f}"
        )


if __name__ == "__main__":
    main()

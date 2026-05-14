"""
Step 1: 模擬SNSデータ生成スクリプト
Ollamaのローカルモデルを使って模擬投稿を生成し、
正解ラベル付きのJSONファイルに保存する。

使い方:
    python generate_data.py
"""

import json
import httpx
import time
import random
from datetime import datetime, timedelta

# ─── 設定 ────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2-vision:latest"   # ollama listで確認したモデル名
OUTPUT_FILE  = "mock_events.json"
TARGET_COUNT = 80  # 生成する投稿数（ノイズ含む）

# ─── 生成シナリオ定義 ─────────────────────────────────────────
# (category, event_id, severity, weather_sev, transport_sev, scenario_desc)
SCENARIOS = [
    ("fire",             "fire_001",     "high",   0.2, 0.5, "豊田市駅前で火災が発生。煙と消防車"),
    ("fire",             "fire_002",     "medium", 0.1, 0.2, "商業施設の裏で小火が発生。煙が少し出ている"),
    ("flood",            "flood_001",    "high",   0.9, 0.4, "大雨で矢作川が増水。周辺道路が冠水"),
    ("flood",            "flood_002",    "medium", 0.7, 0.2, "大雨で住宅地の道路に水が溜まっている"),
    ("traffic_accident", "traffic_001",  "high",   0.1, 0.9, "国道155号で大型トラックと乗用車が衝突"),
    ("traffic_accident", "traffic_002",  "medium", 0.1, 0.6, "交差点で自転車と車が接触事故"),
    ("railway",          "railway_001",  "medium", 0.1, 0.9, "名鉄三河線が人身事故で運転見合わせ"),
    ("noise",            "noise",        "low",    0.0, 0.0, "関係ない日常投稿（ノイズ）"),
]

# シナリオごとの生成数
COUNTS = {
    "fire_001":    12,
    "fire_002":     6,
    "flood_001":   12,
    "flood_002":    6,
    "traffic_001": 10,
    "traffic_002":  6,
    "railway_001": 12,
    "noise":       16,   # ノイズは全体の約20%
}

# 豊田市周辺の位置（ランダムに少しずらす）
BASE_LOCATIONS = {
    "fire_001":    (35.0864, 137.1566, "豊田市駅北口"),
    "fire_002":    (35.0870, 137.1555, "豊田市中心商店街"),
    "flood_001":   (35.0753, 137.1501, "矢作川沿い"),
    "flood_002":   (35.0800, 137.1480, "住宅地（矢作川西側）"),
    "traffic_001": (35.0812, 137.1623, "国道155号 浄水付近"),
    "traffic_002": (35.0840, 137.1590, "市内交差点"),
    "railway_001": (35.0891, 137.1590, "名鉄三河線"),
    "noise":       (35.0860, 137.1560, "豊田市内"),
}

# ─── Ollama呼び出し ────────────────────────────────────────────
def call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.8, "num_predict": 80},
    }
    resp = httpx.post(OLLAMA_URL, json=payload, timeout=120.0)
    resp.raise_for_status()
    return resp.json()["response"].strip()

# ─── プロンプト生成 ───────────────────────────────────────────
def make_prompt(scenario_desc: str, category: str) -> str:
    if category == "noise":
        return """以下のような日常的なSNS投稿を1件だけ日本語で生成してください。
災害や事故とは全く関係ない内容にしてください。
例：食事、天気の感想、買い物、趣味、疲れたなど。
投稿文のみを出力し、説明や鍵かっこは不要です。"""
    else:
        return f"""以下のシナリオについて、現場にいる一般市民がSNSに投稿するような
リアルな短い日本語テキストを1件だけ生成してください。

シナリオ：{scenario_desc}

条件：
- 100文字以内
- 口語・略語・感嘆符を使ってよい
- 場所や状況の具体的な描写を含める
- 投稿文のみを出力し、説明や鍵かっこは不要です"""

# ─── タイムスタンプ生成 ───────────────────────────────────────
def make_timestamp(event_id: str, index: int) -> str:
    base = datetime(2026, 5, 1, 14, 0, 0)
    offsets = {
        "fire_001":    timedelta(hours=0,  minutes=index * 3),
        "fire_002":    timedelta(hours=1,  minutes=index * 5),
        "flood_001":   timedelta(hours=2,  minutes=index * 2),
        "flood_002":   timedelta(hours=2,  minutes=30 + index * 4),
        "traffic_001": timedelta(hours=1,  minutes=30 + index * 4),
        "traffic_002": timedelta(hours=3,  minutes=index * 6),
        "railway_001": timedelta(hours=2,  minutes=20 + index * 3),
        "noise":       timedelta(hours=random.randint(0,3), minutes=random.randint(0,59)),
    }
    t = base + offsets.get(event_id, timedelta(hours=1))
    return t.strftime("%Y-%m-%d %H:%M")

# ─── メイン処理 ──────────────────────────────────────────────
def main():
    all_posts = []
    post_id = 1

    for cat, event_id, severity, weather_sev, transport_sev, scenario_desc in SCENARIOS:
        count = COUNTS[event_id]
        base_lat, base_lng, location = BASE_LOCATIONS[event_id]
        is_noise = (cat == "noise")

        print(f"\n[{event_id}] {scenario_desc} — {count}件生成中...")

        for i in range(count):
            print(f"  {i+1}/{count}件目...", end=" ", flush=True)

            prompt = make_prompt(scenario_desc, cat)
            try:
                text = call_ollama(prompt)
            except Exception as e:
                print(f"エラー: {e}")
                text = scenario_desc  # フォールバック

            # 位置情報を少しずらす（同一事件でも投稿者の位置が異なる）
            lat = base_lat + random.uniform(-0.003, 0.003)
            lng = base_lng + random.uniform(-0.003, 0.003)

            post = {
                "id":                 f"post_{post_id:04d}",
                "timestamp":          make_timestamp(event_id, i),
                "text":               text,
                "location":           location,
                "lat":                round(lat, 6),
                "lng":                round(lng, 6),
                "category":           cat if not is_noise else "noise",
                "severity":           severity,
                "weather_severity":   weather_sev,
                "transport_severity": transport_sev,
                "source_count":       count,
                "event_id":           event_id,
                "is_noise":           is_noise,
            }
            all_posts.append(post)
            post_id += 1
            print("✓")
            time.sleep(0.5)  # Ollamaへの負荷を少し下げる

    # シャッフルして保存（時系列順にしたい場合はコメントアウト）
    random.shuffle(all_posts)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_posts, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完了！{len(all_posts)}件 → {OUTPUT_FILE}")
    print(f"   内訳: ノイズ {COUNTS['noise']}件 / イベント {len(all_posts)-COUNTS['noise']}件")

if __name__ == "__main__":
    main()

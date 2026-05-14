"""主要API関数がデモに必要な値を返すか確認する簡易テスト。"""

import asyncio

from main import analyze_event, get_dashboard, get_system_overview


def main():
    dashboard = get_dashboard(hours=24, category=None)
    assert dashboard["risk_count"] > 0
    assert dashboard["risk_timeline"]
    assert dashboard["top_risk"]["risk_factors"]
    assert dashboard["top_risk"]["confidence_label"]

    overview = get_system_overview()
    assert overview["database"]["event_count"] > 0
    assert overview["pipeline"]

    result = asyncio.run(analyze_event(dashboard["top_risk"]["id"]))
    assert result["analysis"]
    assert result["confidence_score"] >= 0

    print("smoke test ok")


if __name__ == "__main__":
    main()

"""主要API関数が公式情報パイプラインの値を返すか確認する簡易テスト。"""

from config import SIMULATED_DANGEROUS_RATIO, SIMULATED_EVENT_COUNT
from main import get_dashboard, get_system_overview
from simulated_events import build_generation_prompt


def main():
    dashboard = get_dashboard()
    assert "official_observations" in dashboard
    assert "official_summary" in dashboard
    assert "data_summary" in dashboard

    overview = get_system_overview()
    assert overview["pipeline"]
    assert "official_observation_count" in overview["database"]
    assert overview["simulation_defaults"]["event_count"] == SIMULATED_EVENT_COUNT
    assert overview["simulation_defaults"]["dangerous_ratio"] == SIMULATED_DANGEROUS_RATIO

    prompt = build_generation_prompt(count=SIMULATED_EVENT_COUNT, dangerous_ratio=SIMULATED_DANGEROUS_RATIO)
    dangerous_count = round(SIMULATED_EVENT_COUNT * SIMULATED_DANGEROUS_RATIO)
    safe_count = SIMULATED_EVENT_COUNT - dangerous_count
    assert f"events は必ず {SIMULATED_EVENT_COUNT} 件" in prompt
    assert f"低リスク {safe_count} 件、高リスク {dangerous_count} 件" in prompt

    print("smoke test ok")


if __name__ == "__main__":
    main()

"""主要API関数が公式情報パイプラインの値を返すか確認する簡易テスト。"""

from config import OFFICIAL_BACKGROUND_SYNC_ENABLED, SIMULATED_DANGEROUS_RATIO, SIMULATED_EVENT_COUNT
from main import get_dashboard, get_system_overview
from prompts import build_answer_prompt, build_official_normalization_prompt
from simulated_events import build_generation_prompt


def main():
    dashboard = get_dashboard()
    assert "official_observations" in dashboard
    assert "official_summary" in dashboard
    assert "data_summary" in dashboard

    overview = get_system_overview()
    assert overview["pipeline"]
    assert "official_observation_count" in overview["database"]
    assert overview["official_background_sync"]["enabled"] is OFFICIAL_BACKGROUND_SYNC_ENABLED
    assert OFFICIAL_BACKGROUND_SYNC_ENABLED is False
    assert overview["simulation_defaults"]["event_count"] == SIMULATED_EVENT_COUNT
    assert overview["simulation_defaults"]["dangerous_ratio"] == SIMULATED_DANGEROUS_RATIO

    prompt = build_generation_prompt(count=SIMULATED_EVENT_COUNT, dangerous_ratio=SIMULATED_DANGEROUS_RATIO)
    dangerous_count = round(SIMULATED_EVENT_COUNT * SIMULATED_DANGEROUS_RATIO)
    safe_count = SIMULATED_EVENT_COUNT - dangerous_count
    assert f"events は必ず {SIMULATED_EVENT_COUNT} 件" in prompt
    assert f"低リスク {safe_count} 件、高リスク {dangerous_count} 件" in prompt

    normalization_prompt = build_official_normalization_prompt([])
    assert "静岡県・愛知県・岐阜県・三重県" in normalization_prompt

    answer_prompt = build_answer_prompt("三重県の津から船でセントレアにいけますか", [])
    assert "東海地方（静岡県・愛知県・岐阜県・三重県）" in answer_prompt
    assert "航路や便の有無は保存済みデータだけでは確認できません" in answer_prompt
    assert "見合わせを検討" in answer_prompt
    assert "単純な拒答で終わらず" in answer_prompt
    assert "内部用語を使わず" in answer_prompt
    assert "帳票・ラベル形式の行は出力しない" in answer_prompt
    assert "都市安全情報DB" not in answer_prompt

    print("smoke test ok")


if __name__ == "__main__":
    main()

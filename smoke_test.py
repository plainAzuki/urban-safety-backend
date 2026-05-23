"""主要API関数が公式情報パイプラインの値を返すか確認する簡易テスト。"""

from main import get_dashboard, get_system_overview


def main():
    dashboard = get_dashboard()
    assert "official_observations" in dashboard
    assert "official_summary" in dashboard
    assert "data_summary" in dashboard

    overview = get_system_overview()
    assert overview["pipeline"]
    assert "official_observation_count" in overview["database"]

    print("smoke test ok")


if __name__ == "__main__":
    main()

"""FastAPI経由で主要エンドポイントの返却形式を確認する。"""

from fastapi.testclient import TestClient

from main import app


def main():
    client = TestClient(app)
    checks = [
        ("/health", "status"),
        ("/dashboard?hours=24", "risks"),
        ("/system/overview", "pipeline"),
        ("/official/sources", "sources"),
        ("/official/live", "observations"),
        ("/stats", "breakdown"),
    ]
    for path, key in checks:
        resp = client.get(path)
        assert resp.status_code == 200, f"{path}: {resp.status_code}"
        data = resp.json()
        assert key in data, f"{path}: missing {key}"
        print(f"{path} ok")


if __name__ == "__main__":
    main()

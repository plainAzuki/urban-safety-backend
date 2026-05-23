"""SQLite DB を初期化するスクリプト。"""

import sqlite3

from config import DB_FILE
from db import ensure_tables


def main():
    """アプリ本体と同じスキーマ定義を使ってDBを作成する。"""
    conn = sqlite3.connect(DB_FILE)
    ensure_tables(conn)
    conn.commit()
    conn.close()
    print(f"initialized {DB_FILE}")


if __name__ == "__main__":
    main()

"""
    dbクリーナー
"""

import argparse
import sqlite3

from config import DB_FILE
from db import ensure_tables


TABLES_TO_CLEAR = (
    "answer_verifications",
    "official_area_observations",
)


def clear_runtime_tables(vacuum: bool = False) -> dict[str, int]:
    """Delete app runtime records and reset fetch timing state."""
    conn = sqlite3.connect(DB_FILE)
    ensure_tables(conn)
    deleted_counts = {}
    try:
        for table in TABLES_TO_CLEAR:
            cur = conn.execute(f"DELETE FROM {table}")
            deleted_counts[table] = cur.rowcount
        conn.commit()
        if vacuum:
            conn.execute("VACUUM")
    finally:
        conn.close()
    return deleted_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clear Urban Safety runtime DB records and reset the official fetch timer."
    )
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    parser.add_argument("--vacuum", action="store_true", help="Reclaim SQLite file space after deleting rows.")
    args = parser.parse_args()

    if not args.yes:
        answer = input(f"Clear runtime data from {DB_FILE}? Type 'yes' to continue: ")
        if answer.strip().lower() != "yes":
            print("cancelled")
            return

    deleted_counts = clear_runtime_tables(vacuum=args.vacuum)
    for table, count in deleted_counts.items():
        print(f"{table}: deleted {count} rows")
    print("official fetch timer: reset")
    print(f"cleared {DB_FILE}")


if __name__ == "__main__":
    main()

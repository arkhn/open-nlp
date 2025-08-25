#!/usr/bin/env python3
"""
Script to explore the validated_documents.db database
"""

import os
import sqlite3
from pathlib import Path


def explore_database(db_path):
    """Explore the SQLite database and show its contents"""

    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        print(f"🔍 Exploring database: {db_path}")
        print("=" * 50)

        # Get database info
        cursor.execute("SELECT sqlite_version();")
        version = cursor.fetchone()[0]
        print(f"SQLite version: {version}")

        # List all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        if not tables:
            print("📭 No tables found in database")
            return

        print(f"\n📋 Found {len(tables)} table(s):")
        for table in tables:
            table_name = table[0]
            print(f"  • {table_name}")

            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()

            print("    Columns:")
            for col in columns:
                col_id, col_name, col_type, not_null, default_val, pk = col
                pk_marker = " (PK)" if pk else ""
                print(f"      - {col_name}: {col_type}{pk_marker}")

            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            row_count = cursor.fetchone()[0]
            print(f"    Rows: {row_count:,}")

            # Show sample data
            if row_count > 0:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
                sample_rows = cursor.fetchall()

                print("    Sample data:")
                for i, row in enumerate(sample_rows, 1):
                    # Truncate long values for display
                    display_row = []
                    for val in row:
                        if val and len(str(val)) > 100:
                            display_row.append(f"{str(val)[:100]}...")
                        else:
                            display_row.append(str(val))
                    print(f"      Row {i}: {display_row}")

            print()

        conn.close()

    except sqlite3.Error as e:
        print(f"❌ SQLite error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    # Get the database path
    script_dir = Path(__file__).parent
    db_path = script_dir / "validated_documents.db"

    print(f"📁 Script directory: {script_dir}")
    print(f"🗄️  Database path: {db_path}")
    print()

    explore_database(db_path)


if __name__ == "__main__":
    main()

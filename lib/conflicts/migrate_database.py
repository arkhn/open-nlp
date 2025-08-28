import logging
import sqlite3
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_database(db_path: str = "validated_documents.db"):
    """
    Migrate existing database to include timestamp columns

    Args:
        db_path: Path to the database file
    """
    db_file = Path(db_path)

    if not db_file.exists():
        logger.info(f"Database {db_path} does not exist. No migration needed.")
        return

    logger.info(f"Migrating database: {db_path}")

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Check if timestamp columns already exist
            cursor.execute("PRAGMA table_info(validated_documents)")
            columns = [col[1] for col in cursor.fetchall()]

            if "doc1_timestamp" not in columns:
                logger.info("Adding doc1_timestamp column...")
                cursor.execute("ALTER TABLE validated_documents ADD COLUMN doc1_timestamp TEXT")

            if "doc2_timestamp" not in columns:
                logger.info("Adding doc2_timestamp column...")
                cursor.execute("ALTER TABLE validated_documents ADD COLUMN doc2_timestamp TEXT")

            # Check if columns were added successfully
            cursor.execute("PRAGMA table_info(validated_documents)")
            updated_columns = [col[1] for col in cursor.fetchall()]

            logger.info(f"Migration completed. Columns: {updated_columns}")

            # Show sample data
            cursor.execute("SELECT COUNT(*) FROM validated_documents")
            count = cursor.fetchone()[0]
            logger.info(f"Total documents in database: {count}")

            if count > 0:
                cursor.execute("SELECT * FROM validated_documents LIMIT 1")
                sample = cursor.fetchone()
                logger.info(f"Sample document structure: {len(sample)} columns")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


if __name__ == "__main__":
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else "validated_documents.db"
    migrate_database(db_path)
    print("Database migration completed successfully!")

import logging
import re
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from tqdm import tqdm

ROOT = Path(__file__).parent.parent.parent


def main(
    min_docs_per_patient: int = 3,
    n_patients: Optional[int] = 1000,
    mimic_db_path: Path = ROOT / "data" / "mimic.db",
) -> None:
    """Preprocess the MIMIC-III dataset into train and test CSV files.

    Only the discharge summaries are kept, and only for `n_patients` patients with more than
    `min_docs_per_patient` documents. The discharge summaries are then split into sections.

    The script is based on the `mimic.db` SQLite database, which must be downloaded beforehand.

    Args:
        min_docs_per_patient: Minimum number of documents a patient must have to be included.
        n_patients: Number of patients to include in the dataset.
        mimic_db_path: Path to the `mimic.db` SQLite database.
    """
    conn = sqlite3.connect(mimic_db_path)
    logging.basicConfig(level=logging.ERROR)

    # Select the discharge summaries for patients with more than min_docs_per_patient documents
    query = f"""
        SELECT subject_id, row_id, text, chartdate
        FROM noteevents
        WHERE category = 'Discharge summary'
        AND subject_id IN (
            SELECT subject_id
            FROM noteevents
            WHERE category = 'Discharge summary'
            GROUP BY subject_id
            HAVING COUNT(row_id) > {min_docs_per_patient}
        )
        ORDER BY subject_id;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if n_patients:
        df = df[:n_patients]

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        subject_id = row["subject_id"]
        row_id = row["row_id"]
        note = row["text"]
        chartdate = row["chartdate"]

        # Separate the note into sections, with titles and contents
        titles = re.findall(r"^[A-z-\s]+:(?:\n\n|$|\s)", note, flags=re.MULTILINE)
        contents = re.split(r"^[A-z-\s]+:(?:\n\n|$|\s)", note, flags=re.MULTILINE)[1:]
        for title, content in zip(titles, contents):
            title = title.strip().replace(":", "")
            content = content.strip()
            if len(title) < 3 or len(content.split(" ")) < 3:
                # Skip sections with too few words or too short titles
                continue
            records.append(
                {
                    "subject_id": subject_id,
                    "document_id": row_id,
                    "title": title,
                    "content": content,
                    "chartdate": chartdate,
                }
            )

    df = pd.DataFrame.from_records(records)

    # Split into train and test sets, by patient
    subject_ids = df["subject_id"].unique()
    n_train_subjects = int(0.8 * len(subject_ids))
    train_subject_ids = subject_ids[:n_train_subjects]
    test_subject_ids = subject_ids[n_train_subjects:]
    df_train = df[df["subject_id"].isin(train_subject_ids)]
    df_test = df[df["subject_id"].isin(test_subject_ids)]

    df_train.to_csv(ROOT / "data" / "mimoracle_train.csv")
    df_test.to_csv(ROOT / "data" / "mimoracle_test.csv")


if __name__ == "__main__":
    typer.run(main)

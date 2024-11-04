import logging
import os
import re
import shutil
import sqlite3
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def main():
    # Connect to the SQLite database
    conn = sqlite3.connect("./raw/mimic.db")
    logging.basicConfig(level=logging.ERROR)
    # Execute the query and load the result into a pandas DataFrame
    # The char string chat(10)|| '#****#' || char(10) is used as a delimiter
    # for each note concatenated by subject_id in a single string
    query = """
        SELECT subject_id, GROUP_CONCAT(text, char(10)||'#****#'||char(10)) as texts
        FROM noteevents
        WHERE category = 'Discharge summary'
        GROUP BY subject_id;
        """
    df = pd.read_sql_query(query, conn)
    # Iterate over the dataframe and write the notes to files
    for id_row, row in tqdm(df.iterrows(), total=len(df)):
        subject_id = row["subject_id"]
        # Split the notes by the delimiter
        notes = row["texts"].split("\n#****#\n")
        search_phrases = [
            "HISTORY OF PRESENT ILLNESS:",
            "HISTORY OF THE PRESENT ILLNESS:",
            "HPI:",
            "HISTORY:",
            "ADMISSION DIAGNOSIS:",
            "HISTORY OF PRESENT ILLNESS/HOSPITAL COURSE:",
        ]
        # Iterate over the notes and write them to files
        for idx, note in enumerate(notes):
            # pmh is the start of the PMH, hpi is the start of the HPI
            # pmh represents the past medical history, hpi the history of present illness section
            pmh = -1
            hpi = -1
            # Find the start of the HPI
            for phrase in search_phrases:
                hpi = note.upper().find(phrase)
                # If the phrase is found, find the start of the PMH
                if hpi != -1:
                    note = note[hpi + len(phrase) :].strip()
                    pmh = re.search(r"([A-Z]+\s)*[A-Z]{4,}:", note.upper(), re.IGNORECASE)
                    # If the PMH is not found, log a warning, disable by default
                    if pmh is None:
                        logging.warning(
                            "Could not find PMH",
                            subject_id,
                            note[hpi + len(phrase) :],
                        )
                    else:
                        pmh = pmh.start()
                    break
            # If the HPI or PMH is not found, log a warning, disable by default
            if hpi == -1 or pmh == -1:
                logging.warning("Substring not found in note: ", subject_id, note)
            else:
                # If the note is too long, write it to a file
                if note.count("\n") > 5:
                    Path(f"preprocessed/{subject_id}").mkdir(parents=True, exist_ok=True)
                    with open(f"preprocessed/{subject_id}/{idx}.txt", "w") as file:
                        note = note[:pmh].strip()
                        file.write(note)
    # Remove folders with only one file
    for folder in os.listdir("./preprocessed"):
        if len(os.listdir(f"./preprocessed/{folder}")) <= 1:
            shutil.rmtree(f"./preprocessed/{folder}")
    # Close the connection
    conn.close()


if __name__ == "__main__":
    main()

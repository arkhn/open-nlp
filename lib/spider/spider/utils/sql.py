import re
import sqlite3

from spider._path import DATASET_PATH


def extract_sql(message: str, if_first_answer: bool = True) -> str:
    if if_first_answer:
        message = message.split("[/INST]")[-1].replace("\\", "").replace("\n", " ").split(";")[0]
    else:
        message = (
            message.split("[/INST]")[-1].replace("\\", "").replace("\n", " ").split(";")[0] + ";"
        )

    pattern = re.compile(r"select (.*);", re.DOTALL | re.IGNORECASE)
    # find all matches in the message
    matches = pattern.findall(message)
    # check if there are any matches
    if matches:
        # join all matches with a newline character
        sql_code = "\n".join(matches)
        return sql_code.strip().split("\t")[0] + ";"
    else:
        return message.strip().split("\t")[0] + ";"


def is_valid_sql(sql, db_id):
    glob_db = list((DATASET_PATH / "database" / db_id).glob("*.sqlite"))
    if len(glob_db) != 1:
        raise ValueError("Database not found")
    db = str(glob_db[0])
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    try:
        print("response: ", cursor.execute(sql))
        return True, "perfect !"
    except Exception as e:
        return False, e

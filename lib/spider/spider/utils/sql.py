import re
import sqlite3

from spider._path import DATASET_PATH


def extract_sql(message: str, if_first_answer: bool = True) -> str:
    """This function extracts the SQL code from the message.
    first, it removes the schema and question from the message.
    Then, it extracts the SQL code from the message using regex.

    Args:
        message: The message to extract the SQL code from.
        if_first_answer: If this is the first answer,
        then the SQL code is the first part of the message. Defaults to True.

    Returns:
        str: The extracted SQL code.
    """

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


def is_valid_sql(sql: str, db_id: str):
    """This function checks if the SQL code is valid.
    It does so by executing the SQL code on the database.

    Args:
        sql: The SQL code to execute.
        db_id: The database id.

    Returns:
        bool: True if the SQL code is valid, False otherwise.
        str: The error message if the SQL code is invalid.
    """

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

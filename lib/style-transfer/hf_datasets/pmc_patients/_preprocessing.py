import json
from pathlib import Path

from tqdm import tqdm

"""
This script is used to preprocess the PMC Patients dataset.
Pretty simple, it just loads the json file and saves it in a different format
gathering clinical case by title.
"""


def main():
    pmc = json.load(open("./raw/PMC-Patients.json"))
    processed_pmc = {}
    for note in tqdm(pmc, total=len(pmc)):
        patient_uid = note["patient_uid"].split("-")[0]
        patient_sub_uid = note["patient_uid"].split("-")[1]
        text = note["patient"]
        title = note["title"]
        processed_pmc.setdefault(
            patient_uid,
            {
                "title": title,
                "text": [],
            },
        )
        processed_pmc[patient_uid]["text"].append(
            {
                "id": patient_sub_uid,
                "keywords": title,
                "text": text,
            }
        )

    Path(str("preprocessed/")).mkdir(parents=True, exist_ok=True)
    with open("preprocessed/preprocessed-pmc-patients.json", "w") as file:
        file.write(json.dumps(processed_pmc, indent=4))


if __name__ == "__main__":
    main()

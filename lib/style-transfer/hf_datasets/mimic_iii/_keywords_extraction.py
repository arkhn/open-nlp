import json
import os
from pathlib import Path

from quickumls import QuickUMLS
from tqdm import tqdm


def main():
    # define the QuickUMLS installation path in the QUICKUMLS_PATH environment variable
    matcher = QuickUMLS(os.getenv("QUICKUMLS_PATH"))
    Path("keywords_extraction/").mkdir(parents=True, exist_ok=True)
    with open("keywords_extraction/mimic-style-transfer.jsonl", "w") as mimic_style_transfer:
        # Iterate over the preprocessed folder. Each folder is a patient history
        # containing HPI paragraphs.
        for folder in tqdm(os.listdir("./preprocessed"), total=len(os.listdir("./preprocessed"))):
            patient = {folder: []}
            # Iterate over the files in the folder
            for file_id, file in enumerate(os.listdir(f"./preprocessed/{folder}")):
                with open(f"./preprocessed/{folder}/{file}", "r") as f:
                    text = f.read()
                    # Extract the keywords from the text
                    extracted_terms = matcher.match(text, best_match=True, ignore_syntax=False)
                    keywords = [
                        sorted(term, key=lambda x: x["similarity"])[0]["term"]
                        for term in extracted_terms
                    ]
                    # For each patient, append the text and the keywords to the list
                    patient[folder].append(
                        {"id": file_id, "text": text, "keywords": ", ".join(set(keywords))}
                    )

            # Write the patient to the file each line of the jsonl file is a patient
            # with a list of notes and keywords
            mimic_style_transfer.write(json.dumps(patient) + "\n")


if __name__ == "__main__":
    main()

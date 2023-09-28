import json
import os
from itertools import islice
from pathlib import Path

from quickumls import QuickUMLS
from tqdm import tqdm


def main():
    # define the QuickUMLS installation path in the QUICKUMLS_PATH environment variable
    matcher = QuickUMLS(os.getenv("QUICKUMLS_PATH"))
    Path(str("preprocessed")).mkdir(parents=True, exist_ok=True)
    with open("preprocessed/keywords-pmc-patients.jsonl", "w") as mimic_style_transfer:
        # Iterate over the preprocessed processed-pmc-patients.json file
        with open("./preprocessed/preprocessed-pmc-patients.json", "r") as pmc:
            json_pmc = json.load(pmc)
            # each title can represent a list of clinical cases
            for title_id, contents in tqdm(islice(json_pmc.items(), 30000), total=30000):
                title = {title_id: []}
                for content_id, content in enumerate(contents["text"]):
                    text = content
                    extracted_terms = matcher.match(text, best_match=True, ignore_syntax=False)
                    keywords = [
                        sorted(term, key=lambda x: x["similarity"])[0]["term"]
                        for term in extracted_terms
                    ]
                    title[title_id].append(
                        {
                            "id": content_id,
                            "text": text["text"],
                            "keywords": ", ".join(list(set(keywords))),
                            "title": content["keywords"],
                        },
                    )

                # each line of the jsonl file is a title with a list of clinical cases and keywords
                mimic_style_transfer.write(json.dumps(title) + "\n")


if __name__ == "__main__":
    main()

"""
This script is used to preprocess the Yelp dataset.
Pretty simple, it just loads the json file and gathers the comments by user.
"""

import json
from pathlib import Path

from tqdm import tqdm


def main():
    with open("./raw/yelp_academic_dataset_review.json", "r") as yelp:
        processed_yelp_review = {}
        for comment in tqdm(yelp):
            json_comment = json.loads(comment)
            user_id = json_comment["user_id"]
            text = json_comment["text"]
            processed_yelp_review.setdefault(
                user_id,
                [],
            )
            processed_yelp_review[user_id].append(
                {
                    "id": len(json_comment["text"]),
                    "keywords": f'stars: {json_comment["stars"]}, '
                    f'useful: {json_comment["useful"]}, '
                    f'funny: {json_comment["funny"]}, '
                    f'cool: {json_comment["cool"]}',
                    "text": text,
                }
            )

    Path(str("preprocessed/")).mkdir(parents=True, exist_ok=True)
    with open("preprocessed/preprocessed-yelp-review.jsonl", "w") as processed_yelp:
        print("Writing preprocessed yelp review to file ...")
        for user in tqdm(processed_yelp_review, total=len(processed_yelp_review)):
            if len(processed_yelp_review[user]) >= 10:
                json.dump({user: processed_yelp_review[user][:11]}, processed_yelp)
                processed_yelp.write("\n")


if __name__ == "__main__":
    main()

"""
This script is used to extract the keywords from the Yelp dataset. We use the SentiWordNet to get
the words with positive or negative score and add them to the keywords.
"""

import json
from pathlib import Path

import nltk
from nltk.corpus import sentiwordnet
from tqdm import tqdm

nltk.download("sentiwordnet")
nltk.download("stopwords")


def main(size=20000):
    Path(str("preprocessed/")).mkdir(parents=True, exist_ok=True)
    counter = 0
    with open("./preprocessed/keywords-yelp-review.jsonl", "w") as yelp_keywords:
        with open("./preprocessed/preprocessed-yelp-review.jsonl", "r") as yelp:
            # for each user, we iterate over the comments and extract the keywords
            for user in tqdm(yelp, total=size):
                json_user = json.loads(user)
                for comments in json_user.values():
                    for comment in comments:
                        comment_sentiments = []
                        # for each word in the comment,
                        # we check if it is a positive or negative word
                        for word in [
                            word
                            for word in comment["text"].split(" ")
                            if word not in nltk.corpus.stopwords.words("english")
                        ]:
                            sentiments = list(sentiwordnet.senti_synsets(word, pos="a"))
                            if len(sentiments) > 0:
                                for sentiment in sentiments:
                                    if ".a." in sentiment.synset.name() and (
                                        sentiment.pos_score() > 0 or sentiment.neg_score() > 0
                                    ):
                                        comment_sentiments.append(
                                            sentiment.synset.name().split(".")[0]
                                        )
                        # we add the keywords to the comment and stringify the list
                        comment["keywords"] = (
                            comment["keywords"] + ", " + ", ".join(list(set(comment_sentiments)))
                        )
                counter += 1
                yelp_keywords.write(json.dumps(json_user) + "\n")
                if counter >= size:
                    break


if __name__ == "__main__":
    main()

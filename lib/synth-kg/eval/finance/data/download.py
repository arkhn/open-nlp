import argparse
from pathlib import Path

import datasets

DATASETS = [
    # source, destination
    (("pauri32/fiqa-2018", None), "fiqa-2018"),
    (("zeroshot/twitter-financial-news-sentiment", None), "twitter-financial-news-sentiment"),
    (("oliverwang15/news_with_gpt_instructions", None), "news_with_gpt_instructions"),
    (("financial_phrasebank", "sentences_50agree"), "financial_phrasebank-sentences_50agree"),
]


def download(no_cache: bool = False):
    """Downloads all datasets to where the FinGPT library is located."""
    data_dir = Path(__file__).parent

    for src, dest in DATASETS:
        if Path(data_dir / dest).is_dir() and not no_cache:
            print(f"Dataset found at {data_dir / dest}, skipping")
            continue
        dataset = datasets.load_dataset(*src, trust_remote_code=True)
        dataset.save_to_disk(data_dir / dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no_cache",
        default=False,
        required=False,
        type=str,
        help="Redownloads all datasets if set to True",
    )

    args = parser.parse_args()
    download(no_cache=args.no_cache)

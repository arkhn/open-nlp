import json
from pathlib import Path

import pandas as pd


def convert_jsonl_to_parquet(input_path: str, output_path: str):
    data = []
    current_id = 0
    with open(input_path, "r") as f:
        for line in f:
            json_obj = json.loads(line)
            # Extract the list from the first (and only) value in the dictionary
            items = list(json_obj.values())[0]
            for item in items:
                data.append({"id": current_id, "text": item["text"], "keywords": item["keywords"]})
                current_id += 1

    df = pd.DataFrame(data)
    df = df[["id", "text", "keywords"]]
    df.to_parquet(output_path)


def main():
    base_dir = Path("post-processed")
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    # Convert each split
    for split in ["train", "dev", "test"]:
        input_path = base_dir / f"{split}.jsonl"
        output_path = output_dir / f"{split}.parquet"
        convert_jsonl_to_parquet(str(input_path), str(output_path))


if __name__ == "__main__":
    main()

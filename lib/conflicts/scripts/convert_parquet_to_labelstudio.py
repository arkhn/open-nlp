import argparse
import json

import pandas as pd


def find_text_positions(text, excerpt):
    if not excerpt or pd.isna(excerpt):
        return None, None

    start_pos = text.find(excerpt)
    if start_pos == -1:
        return None, None

    end_pos = start_pos + len(excerpt)
    return start_pos, end_pos


def convert_row_to_labelstudio(row):
    data_item = {
        "data": {
            "doc_1": row["modified_doc1_text"],
            "doc_2": row["modified_doc2_text"],
            "timestamp_1": row["doc1_timestamp"],
            "timestamp_2": row["doc2_timestamp"],
        },
        "annotations": [{"result": []}],
    }

    if pd.notna(row["modified_excerpt_1"]) and row["modified_excerpt_1"].strip():
        start_pos, end_pos = find_text_positions(
            row["modified_doc1_text"], row["modified_excerpt_1"]
        )
        if start_pos is not None:
            data_item["annotations"][0]["result"].append(
                {
                    "from_name": "labels_doc1",
                    "to_name": "doc_1",
                    "type": "labels",
                    "value": {
                        "start": start_pos,
                        "end": end_pos,
                        "text": row["modified_excerpt_1"],
                        "labels": ["Conflict"],
                    },
                }
            )

    if pd.notna(row["modified_excerpt_2"]) and row["modified_excerpt_2"].strip():
        start_pos, end_pos = find_text_positions(
            row["modified_doc2_text"], row["modified_excerpt_2"]
        )
        if start_pos is not None:
            data_item["annotations"][0]["result"].append(
                {
                    "from_name": "labels_doc2",
                    "to_name": "doc_2",
                    "type": "labels",
                    "value": {
                        "start": start_pos,
                        "end": end_pos,
                        "text": row["modified_excerpt_2"],
                        "labels": ["Conflict"],
                    },
                }
            )

    return data_item


def convert_parquet_to_labelstudio(parquet_path, output_path):
    df = pd.read_parquet(parquet_path)

    labelstudio_data = []
    for _, row in df.iterrows():
        data_item = convert_row_to_labelstudio(row)
        labelstudio_data.append(data_item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labelstudio_data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_parquet")
    parser.add_argument("output_json")

    args = parser.parse_args()

    convert_parquet_to_labelstudio(args.input_parquet, args.output_json)


if __name__ == "__main__":
    main()

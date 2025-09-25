import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List


def filter_annotations(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep best conflict pair + random annotation."""
    filtered_data = []

    for item in data:
        filtered_item = {"data": item["data"].copy()}

        if "annotations" in item and item["annotations"]:
            annotations = item["annotations"][0]["result"]

            # Sort by score (descending) and validity
            sorted_annotations = sorted(
                annotations,
                key=lambda ann: (-ann.get("moderator_score", 0), -ann.get("is_valid", False)),
            )

            selected_annotations = []

            # 1. Find best conflict pair
            for ann in sorted_annotations:
                if ann.get("to_name") == "doc_1":
                    # Find matching doc_2
                    doc1_from_name = ann.get("from_name", "")
                    matching_doc2_name = doc1_from_name.replace("_doc1", "_doc2")

                    for ann2 in sorted_annotations:
                        if (
                            ann2.get("to_name") == "doc_2"
                            and ann2.get("from_name") == matching_doc2_name
                        ):
                            selected_annotations = [ann, ann2]
                            break
                    break  # Stop after finding first complete pair

            # 2. Add random annotation from remaining
            remaining = [ann for ann in sorted_annotations if ann not in selected_annotations]
            if remaining:
                # Prefer low-score annotations
                low_score = [ann for ann in remaining if ann.get("moderator_score", 0) < 4]
                selected_annotations.append(random.choice(low_score or remaining))

            filtered_item["annotations"] = [{"result": selected_annotations}]
        else:
            filtered_item["annotations"] = []

        filtered_data.append(filtered_item)

    return filtered_data


def process_file(input_path: str, output_path: str) -> None:
    """Process a single JSON file."""
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)

    print(f"Processing: {input_file} → {output_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} document pairs")

    # Filter and save
    filtered_data = filter_annotations(data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully processed {len(filtered_data)} document pairs")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Filter conflict data")
    parser.add_argument("input_files", nargs="+", help="Input JSON file(s)")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    for input_file in args.input_files:
        if args.output and len(args.input_files) == 1:
            output_file = args.output
        else:
            input_path = Path(input_file)
            output_file = input_path.parent / f"{input_path.stem}_selected{input_path.suffix}"

        process_file(input_file, output_file)
        print()


if __name__ == "__main__":
    main()

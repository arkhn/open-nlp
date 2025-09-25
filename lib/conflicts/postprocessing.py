import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class PostprocessingConfig:
    """Configuration for postprocessing parameters."""

    def __init__(
        self,
        low_score_threshold: float = 4.0,
        sort_by_score: bool = True,
        sort_by_validity: bool = True,
    ):
        self.low_score_threshold = low_score_threshold
        self.sort_by_score = sort_by_score
        self.sort_by_validity = sort_by_validity


def extract_annotations(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract and validate annotations from a data item."""
    if not item.get("annotations") or not item["annotations"]:
        return []

    annotations = item["annotations"][0].get("result", [])
    if not isinstance(annotations, list):
        return []

    return annotations


def create_annotation_lookup(annotations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Create efficient lookup dictionaries for doc_1 and doc_2 annotations."""
    doc1_lookup = {}
    doc2_lookup = {}

    for ann in annotations:
        to_name = ann.get("to_name", "")
        from_name = ann.get("from_name", "")

        if to_name == "doc_1":
            doc1_lookup[from_name] = ann
        elif to_name == "doc_2":
            doc2_lookup[from_name] = ann

    return doc1_lookup, doc2_lookup


def find_matching_pair(
    doc1_ann: Dict[str, Any], doc2_lookup: Dict[str, Dict[str, Any]]
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Find matching doc_2 annotation for a given doc_1 annotation."""
    from_name = doc1_ann.get("from_name", "")

    # Try different naming patterns for robustness
    possible_doc2_names = [
        from_name.replace("_doc1", "_doc2"),
        from_name.replace("doc1", "doc2"),
        from_name + "_doc2"
        if not from_name.endswith("_doc1")
        else from_name.replace("_doc1", "_doc2"),
    ]

    for doc2_name in possible_doc2_names:
        if doc2_name in doc2_lookup:
            return doc1_ann, doc2_lookup[doc2_name]

    return None


def create_data_item(
    original_item: Dict[str, Any], annotation_pair: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Create a new data item from original item and annotation pair."""
    doc1_ann = annotation_pair[0]  # Use doc_1 annotation for metadata

    # Extract moderator metadata
    moderator_fields = [
        "moderator_score",
        "moderator_reasoning",
        "conflict_type",
        "clinical_plausibility_score",
        "temporal_appropriateness_score",
        "clinical_significance_score",
        "is_valid",
        "retry_attempt",
    ]

    moderator_data = {
        field: doc1_ann.get(
            field,
            0
            if field.endswith("_score")
            else ""
            if field == "moderator_reasoning" or field == "conflict_type"
            else False
            if field == "is_valid"
            else 0,
        )
        for field in moderator_fields
    }

    return {
        "data": {**original_item["data"], **moderator_data},
        "annotations": [{"result": annotation_pair}],
    }


def find_best_pair(
    annotations: List[Dict[str, Any]],
    doc1_lookup: Dict[str, Dict[str, Any]],
    doc2_lookup: Dict[str, Dict[str, Any]],
    config: PostprocessingConfig,
) -> Optional[List[Dict[str, Any]]]:
    """Find the best conflict pair based on scoring criteria."""
    if not doc1_lookup:
        return None

    # Sort doc_1 annotations by quality
    doc1_annotations = list(doc1_lookup.values())
    if config.sort_by_score and config.sort_by_validity:
        doc1_annotations.sort(
            key=lambda ann: (-ann.get("moderator_score", 0), -ann.get("is_valid", False))
        )
    elif config.sort_by_score:
        doc1_annotations.sort(key=lambda ann: -ann.get("moderator_score", 0))
    elif config.sort_by_validity:
        doc1_annotations.sort(key=lambda ann: -ann.get("is_valid", False))

    # Find first valid pair
    for doc1_ann in doc1_annotations:
        pair = find_matching_pair(doc1_ann, doc2_lookup)
        if pair:
            return list(pair)

    return None


def find_random_pair(
    annotations: List[Dict[str, Any]],
    doc1_lookup: Dict[str, Dict[str, Any]],
    doc2_lookup: Dict[str, Dict[str, Any]],
    best_pair: Optional[List[Dict[str, Any]]],
    config: PostprocessingConfig,
) -> Optional[List[Dict[str, Any]]]:
    """Find a random conflict pair, preferring low-score annotations."""
    if not doc1_lookup:
        return None

    # Get remaining doc_1 annotations (excluding best pair)
    remaining_doc1 = list(doc1_lookup.values())
    if best_pair:
        best_doc1 = best_pair[0]
        remaining_doc1 = [ann for ann in remaining_doc1 if ann != best_doc1]

    if not remaining_doc1:
        return None

    # Prefer low-score annotations for variety
    low_score_annotations = [
        ann for ann in remaining_doc1 if ann.get("moderator_score", 0) < config.low_score_threshold
    ]

    selected_doc1 = random.choice(low_score_annotations or remaining_doc1)
    pair = find_matching_pair(selected_doc1, doc2_lookup)

    return list(pair) if pair else None


def filter_annotations(
    data: List[Dict[str, Any]], config: Optional[PostprocessingConfig] = None
) -> List[Dict[str, Any]]:
    """Create separate data entries for best pair and random pair."""
    if config is None:
        config = PostprocessingConfig()

    filtered_data = []

    for item in data:
        annotations = extract_annotations(item)
        if not annotations:
            continue

        # Create efficient lookups
        doc1_lookup, doc2_lookup = create_annotation_lookup(annotations)

        # Find pairs
        best_pair = find_best_pair(annotations, doc1_lookup, doc2_lookup, config)
        random_pair = find_random_pair(annotations, doc1_lookup, doc2_lookup, best_pair, config)

        # Create output items
        if best_pair:
            filtered_data.append(create_data_item(item, best_pair))

        if random_pair:
            filtered_data.append(create_data_item(item, random_pair))

    return filtered_data


def process_file(
    input_path: str,
    output_path: str,
    config: Optional[PostprocessingConfig] = None,
    verbose: bool = False,
) -> None:
    """Process a single JSON file."""
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)

    if verbose:
        print(f"Processing: {input_file} → {output_file}")

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading file '{input_path}': {e}")
        sys.exit(1)

    if not isinstance(data, list):
        print(f"Error: Expected list of items, got {type(data)}")
        sys.exit(1)

    if verbose:
        print(f"Loaded {len(data)} document pairs")

    # Filter and save
    try:
        filtered_data = filter_annotations(data, config)
    except Exception as e:
        print(f"Error processing data: {e}")
        sys.exit(1)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"Error writing file '{output_path}': {e}")
        sys.exit(1)

    if verbose:
        print(f"Successfully processed {len(filtered_data)} document pairs")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Filter conflict data")
    parser.add_argument("input_files", nargs="+", help="Input JSON file(s)")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--low-score-threshold",
        type=float,
        default=4.0,
        help="Threshold for low-score annotations (default: 4.0)",
    )
    parser.add_argument(
        "--no-sort-by-score", action="store_true", help="Disable sorting by moderator score"
    )
    parser.add_argument(
        "--no-sort-by-validity", action="store_true", help="Disable sorting by validity"
    )

    args = parser.parse_args()

    # Create configuration
    config = PostprocessingConfig(
        low_score_threshold=args.low_score_threshold,
        sort_by_score=not args.no_sort_by_score,
        sort_by_validity=not args.no_sort_by_validity,
    )

    for input_file in args.input_files:
        if args.output and len(args.input_files) == 1:
            output_file = args.output
        else:
            input_path = Path(input_file)
            output_file = input_path.parent / f"{input_path.stem}_selected{input_path.suffix}"

        process_file(input_file, output_file, config, args.verbose)
        if args.verbose:
            print()


if __name__ == "__main__":
    main()

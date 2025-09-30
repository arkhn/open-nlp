import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


def get_git_sha() -> str:
    """Get current git SHA-1 hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()[:7]  # Short SHA
    except subprocess.CalledProcessError:
        return "nogit"


def get_pair_id(item: Dict[str, Any]) -> str:
    """Extract unique pair ID from data item."""
    data = item.get("data", {})
    doc1 = data.get("doc_1", "")
    doc2 = data.get("doc_2", "")

    # Create unique identifier from document content only
    content = f"{doc1}{doc2}"
    return hashlib.md5(content.encode()).hexdigest()[:8]


def calculate_average_score(item: Dict[str, Any]) -> float:
    """Calculate average of the three individual scores."""
    clinical_plausibility = item.get("clinical_plausibility_score", 0)
    temporal_appropriateness = item.get("temporal_appropriateness_score", 0)
    clinical_significance = item.get("clinical_significance_score", 0)

    scores = [clinical_plausibility, temporal_appropriateness, clinical_significance]
    valid_scores = [s for s in scores if s is not None and s != 0]

    return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0


def load_and_process_files(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Load and aggregate data from multiple JSON files."""
    all_data = []

    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_data.extend(data if isinstance(data, list) else [data])
            print(f"Loaded {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return all_data


def create_splits(data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Create two splits based on criteria."""
    used_pair_ids = set()
    best_split = []
    low_score_split = []

    # Split 1: Best propositions
    for item in data:
        pair_id = get_pair_id(item)
        is_best = item.get("best_conflict", False) or item.get("is_best", False)

        if is_best and pair_id not in used_pair_ids:
            best_split.append(item)
            used_pair_ids.add(pair_id)

    # Split 2: Low-scored propositions
    for item in data:
        pair_id = get_pair_id(item)
        avg_score = calculate_average_score(item)

        if avg_score < 4.0 and pair_id not in used_pair_ids:
            low_score_split.append(item)
            used_pair_ids.add(pair_id)

    print(f"Best split: {len(best_split)} items")
    print(f"Low score split: {len(low_score_split)} items")

    return best_split, low_score_split


def create_output_filename() -> str:
    """Create output filename with git SHA and date."""
    git_sha = get_git_sha()
    date_str = datetime.now().strftime("%d%m%Y")
    return f"{git_sha}_{date_str}"


def create_metadata(
    input_files: List[str],
    best_count: int,
    low_score_count: int,
    total_count: int,
    output_filename: str,
) -> str:
    """Create metadata content."""
    git_sha = get_git_sha()
    timestamp = datetime.now().isoformat()

    metadata = f"""# Dataset Postprocessing Metadata

    ## Postprocessed Information
    - **Postprocessed Date**: {timestamp}
    - **Git SHA**: {git_sha}
    - **Output File**: {output_filename}.json

    ## Input Files
    """

    for file_path in input_files:
        metadata += f"- `{Path(file_path).name}`\n"

    metadata += f"""
    ## Postprocessed Results
    - **Total Input Items**: {total_count}
    - **Best Propositions Split**: {best_count} items (unique pair IDs)
    - **Low Score Split**: {low_score_count} items (average score < 4.0, unique pair IDs)
    - **Final Dataset Size**: {best_count + low_score_count} items
    """

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Post-process conflict detection dataset files")
    parser.add_argument("files", nargs="+", help="JSON files to process")
    args = parser.parse_args()

    # Load and process data
    print(f"Processing {len(args.files)} files...")
    all_data = load_and_process_files(args.files)

    print(f"Total items loaded: {len(all_data)}")

    # Create splits
    best_split, low_score_split = create_splits(all_data)
    merged_data = best_split + low_score_split

    # Create output directory
    output_dir = Path("postprocessed")
    output_dir.mkdir(exist_ok=True)

    # Create output filename
    output_filename = create_output_filename()

    # Write JSON output
    json_path = output_dir / f"{output_filename}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    # Write metadata
    metadata_content = create_metadata(
        args.files, len(best_split), len(low_score_split), len(all_data), output_filename
    )

    md_path = output_dir / f"{output_filename}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(metadata_content)

    print("Output files created:")
    print(f"  - {json_path}")
    print(f"  - {md_path}")
    print(
        f"\nFinal dataset: {len(merged_data)} items ({len(best_split)} best"
        f"{len(low_score_split)} low score)"
    )

    return 0


if __name__ == "__main__":
    exit(main())

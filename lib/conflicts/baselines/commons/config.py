import os
import sys
from pathlib import Path

# Conflict type mapping - single source of truth
CONFLICT_TYPES = {
    "opposition": 0,
    "anatomical": 1,
    "value": 2,
    "contraindication": 3,
    "comparison": 4,
    "descriptive": 5,
}

# Conflict type labels in order
CONFLICT_LABELS = list(CONFLICT_TYPES.keys())

# Reverse mapping
ID_TO_LABEL = {idx: label for label, idx in CONFLICT_TYPES.items()}


def setup_paths():
    """Setup Python path for imports."""
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)


def get_conflicts_data_path(data_path: str = "processed/_12092025.json") -> Path:
    """
    Get the full path to the conflicts data file.

    Args:
        data_path: Relative path to data file

    Returns:
        Full path to data file
    """
    conflicts_dir = Path(__file__).parent.parent.parent
    return conflicts_dir / data_path

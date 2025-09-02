#!/usr/bin/env python3
import logging

import hydra
from conflicts.core.pipeline import Pipeline
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../", config_name="config")
def main(cfg: DictConfig) -> None:
    """Show pipeline statistics"""

    pipeline = Pipeline(cfg)
    stats = pipeline.get_pipeline_statistics()

    print(f"Validated documents in database: {stats['validated_documents']}")
    print(f"Dataset documents: {stats['dataset_statistics']['total_documents']}")
    print(f"Unique subjects: {stats['dataset_statistics']['unique_subjects']}")
    print(f"Available categories: {', '.join(stats['dataset_statistics']['sample_categories'])}")

    print("\n=== AGENTS ===")
    print(f"Doctor Agent: {stats['agents']['doctor']['name']}")
    print(f"Editor Agent: {stats['agents']['editor']['name']}")
    print(
        f"Moderator Agent: {stats['agents']['moderator']['name']} "
        f"(min score: {stats['agents']['moderator']['min_score']})"
    )

    print("\n=== CONFIGURATION ===")
    print(f"Max retries: {stats['configuration']['max_retries']}")
    print(f"Dataset: {stats['configuration']['dataset_path']}")


if __name__ == "__main__":
    main()

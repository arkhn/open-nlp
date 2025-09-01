from typing import List, Optional

import typer
from hydra import compose, initialize  # <- API Hydra (pas de décorateur)
from omegaconf import DictConfig
from pipeline import Pipeline

app = typer.Typer(help="Clinical Document Conflict Pipeline")


def load_cfg(overrides: Optional[List[str]] = None) -> DictConfig:
    """
    Compose Hydra config without hijacking CLI.
    Example overrides: ["pipeline.dataset_size=100", "agents.editor.model=gpt-4o"]
    """
    with initialize(version_base=None, config_path="."):
        cfg = compose(config_name="config", overrides=overrides or [])
    return cfg


@app.command()
def batch(
    overrides: List[str] = typer.Argument(
        None,
        help="Hydra overrides, ex: pipeline.dataset_size=100 agents.editor.model=gpt-4o",
    )
) -> None:
    """Process a batch of document pairs using Hydra configuration"""
    cfg = load_cfg(overrides)
    pipeline = Pipeline(cfg)

    size = cfg.pipeline.dataset_size
    print(f"Processing batch of {size} document pairs...")

    result = pipeline.execute(
        dataset_size=size,
        category_filter=None,
    )

    print(f"Total pairs processed: {result['total_pairs']}")
    print(f"Successful: {result['successful']}")
    print(f"Failed: {result['failed']}")
    print(f"Success rate: {result['success_rate']:.1f}%")
    print(f"Total processing time: {result['total_processing_time']:.2f}s")

    pipeline.db_manager.save_to_parquet()
    print("Results saved to Parquet file.")


@app.command()
def stats(
    overrides: List[str] = typer.Argument(
        None,
        help="Hydra overrides, ex: database.path=data/db.parquet",
    )
) -> None:
    """Show pipeline statistics"""
    cfg = load_cfg(overrides)
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
    print(f"Database: {stats['configuration']['database_path']}")


if __name__ == "__main__":
    app()

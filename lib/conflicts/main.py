from typing import List, Optional

import typer
from pipeline import Pipeline

app = typer.Typer(help="Clinical Document Conflict Pipeline")


@app.command()
def batch(
    size: int = typer.Option(5, help="Batch size"),
    categories: Optional[List[str]] = typer.Option(
        None, help='Filter by document categories (e.g. "Discharge summary" "Progress note")'
    ),
    max_retries: int = typer.Option(3, help="Maximum retry attempts for validation failures"),
    min_score: int = typer.Option(70, "--min-score", help="Minimum validation score required"),
):
    """Process a batch of document pairs"""
    pipeline = Pipeline(
        max_retries=max_retries,
        min_validation_score=min_score,
    )

    print(f"Processing batch of {size} document pairs...")

    result = pipeline.process_batch(
        batch_size=size,
        category_filter=categories,
    )

    print(f"Total pairs processed: {result['total_pairs']}")
    print(f"Successful: {result['successful']}")
    print(f"Failed: {result['failed']}")
    print(f"Success rate: {result['success_rate']:.1f}%")
    print(f"Total processing time: {result['total_processing_time']:.2f}s")


@app.command()
def stats():
    """Show pipeline statistics"""
    pipeline = Pipeline()
    stats = pipeline.get_pipeline_statistics()

    print(f"Validated documents in database: {stats['validated_documents']}")
    print(f"Dataset documents: {stats['dataset_statistics']['total_documents']}")
    print(f"Unique subjects: {stats['dataset_statistics']['unique_subjects']}")
    print(f"Available categories: {', '.join(stats['dataset_statistics']['sample_categories'])}")

    print("\n=== AGENTS ===")
    print(f"Doctor Agent: {stats['agents']['doctor']['name']}")
    print(f"Editor Agent: {stats['agents']['editor']['name']}")
    print(
        f"Moderator Agent: {stats['agents']['moderator']['name']} \
            (min score: {stats['agents']['moderator']['min_validation_score']})"
    )

    print("\n=== CONFIGURATION ===")
    print(f"Max retries: {stats['configuration']['max_retries']}")
    print(f"Database: {stats['configuration']['database_path']}")


if __name__ == "__main__":
    app()

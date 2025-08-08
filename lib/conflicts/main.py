import argparse
import sys

from pipeline import ClinicalConflictPipeline


def main():
    parser = argparse.ArgumentParser(description="Clinical Document Conflict Pipeline")

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process batch command
    batch_parser = subparsers.add_parser("batch", help="Process a batch of document pairs")
    batch_parser.add_argument("--size", type=int, default=5, help="Batch size (default: 5)")
    batch_parser.add_argument(
        "--same-subject", action="store_true", help="Try to pair documents from the same subject"
    )
    batch_parser.add_argument(
        "--categories",
        nargs="+",
        help='Filter by document categories (e.g. "Discharge summary" "Progress note")',
    )
    batch_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for validation failures (default: 3)",
    )
    batch_parser.add_argument(
        "--min-score", type=int, default=70, help="Minimum validation score required (default: 70)"
    )

    # Test conflict type command
    test_parser = subparsers.add_parser("test-conflict", help="Test a specific conflict type")
    test_parser.add_argument(
        "conflict_type", help="Conflict type to test (e.g. opposition, anatomical, value)"
    )
    test_parser.add_argument(
        "--count", type=int, default=3, help="Number of document pairs to test (default: 3)"
    )

    # Statistics command
    subparsers.add_parser("stats", help="Show pipeline statistics")

    # List conflicts command
    subparsers.add_parser("list-conflicts", help="List available conflict types")

    # Database command
    subparsers.add_parser("db-info", help="Show database information")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        # Commands that don't require API key
        if args.command == "db-info":
            from pathlib import Path

            from base import DatabaseManager

            db_manager = DatabaseManager()
            db_count = db_manager.get_validated_documents_count()
            db_path = db_manager.db_path

            print("=== DATABASE INFORMATION ===")
            print(f"Database path: {db_path}")
            print(f"Database exists: {Path(db_path).exists()}")
            print(f"Validated documents: {db_count}")

            if Path(db_path).exists():
                db_size = Path(db_path).stat().st_size
                print(f"Database size: {db_size / 1024:.1f} KB")
            return

        # Initialize pipeline
        pipeline = ClinicalConflictPipeline(
            max_retries=getattr(args, "max_retries", 3),
            min_validation_score=getattr(args, "min_score", 70),
        )

        if args.command == "batch":
            print(f"Processing batch of {args.size} document pairs...")

            result = pipeline.process_batch(
                batch_size=args.size,
                same_subject=args.same_subject,
                category_filter=args.categories,
            )

            print("\n=== BATCH PROCESSING RESULTS ===")
            print(f"Total pairs processed: {result['total_pairs']}")
            print(f"Successful: {result['successful']}")
            print(f"Failed: {result['failed']}")
            print(f"Success rate: {result['success_rate']:.1f}%")
            print(f"Total processing time: {result['total_processing_time']:.2f}s")
            print(f"Average per pair: {result['average_processing_time']:.2f}s")

            # Show detailed results for failed cases
            failed_results = [r for r in result["results"] if not r["success"]]
            if failed_results:
                print(f"\n=== FAILED CASES ({len(failed_results)}) ===")
                for failed in failed_results:
                    print(f"Pair {failed['pair_id']}: {failed.get('error', 'Unknown error')}")

        elif args.command == "test-conflict":
            print(f"Testing conflict type: {args.conflict_type}")

            result = pipeline.test_single_conflict_type(args.conflict_type, args.count)

            if not result["success"]:
                print(f"Error: {result['error']}")
                return

            print("\n=== CONFLICT TYPE TEST RESULTS ===")
            print(f"Conflict type tested: {result['conflict_type_tested']}")
            print(f"Document pairs tested: {result['total_pairs']}")
            print(f"Successful validations: {result['successful_validations']}")
            print(f"Success rate: {result['success_rate']:.1f}%")

            for test_result in result["results"]:
                status = "PASS" if test_result.get("validation_success") else "FAIL"
                score = test_result.get("validation_score", 0)
                print(f"  {test_result['pair_id']}: {status} (Score: {score}/100)")

        elif args.command == "stats":
            stats = pipeline.get_pipeline_statistics()

            print("=== PIPELINE STATISTICS ===")
            print(f"Validated documents in database: {stats['validated_documents']}")
            print(f"Dataset documents: {stats['dataset_statistics']['total_documents']}")
            print(f"Unique subjects: {stats['dataset_statistics']['unique_subjects']}")
            print(f"Categories: {', '.join(stats['dataset_statistics']['sample_categories'])}")

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

        elif args.command == "list-conflicts":
            # Use doctor agent to get conflict types
            conflict_types = pipeline.doctor_agent.list_all_conflict_types()

            print("=== AVAILABLE CONFLICT TYPES ===")
            for key, info in conflict_types.items():
                print(f"\n{key.upper()}: {info['name']}")
                print(f"Description: {info['description']}")
                print("Examples:")
                for example in info["examples"][:2]:  # Show first 2 examples
                    print(f"  - {example}")
                if len(info["examples"]) > 2:
                    print(f"  ... and {len(info['examples']) - 2} more")

        elif args.command == "db-info":
            db_count = pipeline.db_manager.get_validated_documents_count()
            db_path = pipeline.db_manager.db_path

            print("=== DATABASE INFORMATION ===")
            print(f"Database path: {db_path}")
            print(f"Database exists: {Path(db_path).exists()}")
            print(f"Validated documents: {db_count}")

            if Path(db_path).exists():
                db_size = Path(db_path).stat().st_size
                print(f"Database size: {db_size / 1024:.1f} KB")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

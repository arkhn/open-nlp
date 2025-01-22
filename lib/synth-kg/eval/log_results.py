import wandb
import argparse
import json
from typing import Dict, Any


def log_results_to_run(run_id: str, results: Dict[str, Any]) -> None:
    """Log results to an existing wandb run."""
    api = wandb.Api()
    try:
        run = api.run(run_id)
        with wandb.init(id=run.id, project="synth-kg", resume="must") as run:
            wandb.log(results)
    except Exception as e:
        print(f"Error logging to run {run_id}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Log results to a specific W&B run")
    parser.add_argument(
        "--run_id", type=str, required=True, help="W&B run ID"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="JSON string or path to JSON file containing results to log",
    )

    args = parser.parse_args()

    # Parse results
    try:
        if args.results.endswith(".json"):
            with open(args.results, "r") as f:
                results = json.load(f)
        else:
            results = json.loads(args.results)
    except Exception as e:
        print(f"Error parsing results: {str(e)}")
        return

    # Log results
    log_results_to_run(args.run_id, results)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Synth-KG CLI - A clean and beautiful command-line interface
"""
import logging
import subprocess
from typing import List

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

# Setup rich console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("synth-kg")

# Create main app
app = typer.Typer(
    help="💡 Synth-KG: Command-line tools",
    add_completion=True,
)

# Create subcommands for better organization
jz_app = typer.Typer(help="Jean Zay HPC cluster commands")
data_app = typer.Typer(help="Dataset management commands")
wandb_app = typer.Typer(help="Weights & Biases commands")

app.add_typer(jz_app, name="jz")
app.add_typer(data_app, name="data")
app.add_typer(wandb_app, name="wandb")


def run_command(command: str) -> str:
    """Execute a shell command and return its output."""
    logger.info(f"Running: [bold cyan]{command}[/bold cyan]")
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise Exception(f"Command failed: [bold red]{command}: {stderr}[/bold red]")

    return stdout.strip()


def run_sbatch(script_path: str, args: List[str] = []) -> str:
    """Run an sbatch command with optional arguments."""
    args = args or []
    args_str = " ".join(args) if args else ""
    command = f"sbatch {script_path} {args_str}".strip()
    return run_command(command)


@jz_app.command("download-resources")
def jz_download_resources():
    """Download resources for the project."""
    run_sbatch("launch/jz/download_resources.slurm")


@data_app.command("upload")
def upload_datasets():
    """Upload datasets to Amazon S3."""
    logger.info("Uploading datasets to S3")
    run_command("aws s3 sync ./datasets s3://synth-kg-dataset")
    logger.info("✅ Upload complete")


@data_app.command("download")
def download_datasets():
    """Download datasets from Amazon S3."""
    logger.info("Downloading datasets from S3")
    run_command("aws s3 sync s3://synth-kg-datasets ./datasets")
    logger.info("✅ Download complete")


@wandb_app.command("sync")
def wandb_sync(
    path: str = typer.Option(
        "/lustre/fswork/projects/rech/lch/ufw96he/.wandb/wandb/offline*",
        help="Path to W&B offline files",
    )
):
    """Sync local Weights & Biases data to cloud."""
    logger.info(f"Syncing W&B data from: [bold]{path}[/bold]")
    run_command(f"wandb sync --include-offline --include-synced {path}")
    logger.info("✅ Sync complete")


@wandb_app.command("clean")
def clean_wandb_run(
    path: str = typer.Option(
        "/lustre/fswork/projects/rech/lch/ufw96he/.wandb/wandb/offline*",
        help="Path to W&B offline files",
    )
):
    """Remove local Weights & Biases run files."""
    logger.info(f"Cleaning W&B run files from: [bold]{path}[/bold]")
    run_command(f"rm -r {path}")
    logger.info("✅ Cleanup complete")


@app.command()
def info():
    """Display information about available commands and structure."""
    console.print(
        Panel.fit(
            "[bold cyan]Synth-KG CLI[/bold cyan]",
            title="About",
        )
    )

    groups = Table(title="Command Groups")
    groups.add_column("Group", style="green")
    groups.add_column("Description", style="blue")
    groups.add_row("jz", "Jean Zay HPC cluster commands")
    groups.add_row("data", "Dataset management commands")
    groups.add_row("wandb", "Weights & Biases commands")
    console.print(groups)
    console.print("\nUse [bold green]--help[/bold green] with any command for more information")


if __name__ == "__main__":
    app()

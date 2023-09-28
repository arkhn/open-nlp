import pathlib
from pathlib import Path

import typer
from huggingface_hub import HfApi

app = typer.Typer()
api = HfApi()


@app.command()
def upload_file(file_path: str) -> None:
    """Upload a file to the Hugging Face Hub.

    Args:
        file_path: Path to the file to upload.
    """
    filename = Path(file_path).name
    repo_id = "bio-datasets/e3c-llm"
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=pathlib.PurePath(file_path).name,
        repo_id=repo_id,
        repo_type="dataset",
    )
    typer.echo(f"Successfully uploaded file {file_path} to {repo_id}/{filename}")


@app.command()
def upload_folder(folder_path: str) -> None:
    """Upload a folder to the Hugging Face Hub.

    Args:
        folder_path: Path to the folder to upload.
    """
    repo_id = "bio-datasets/e3c-llm"
    api.upload_folder(
        folder_path=folder_path,
        path_in_repo=pathlib.PurePath(folder_path).name,
        repo_id=repo_id,
        repo_type="dataset",
    )
    typer.echo(f"Successfully uploaded folder {folder_path} to {repo_id}/{folder_path}")


if __name__ == "__main__":
    app()

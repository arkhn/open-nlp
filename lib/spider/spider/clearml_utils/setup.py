import os
import warnings
from pathlib import Path
from typing import Union

from dotenv import load_dotenv

from spider._path import _ROOT

# Names of the environment variables which must be set
CLEARML_VAR_NAMES: set[str] = {
    "CLEARML_SERVER",
    "CLEARML_USERNAME",
    "CLEARML_PASSWORD",
    "MINIO_SERVER",
    "MINIO_USERNAME",
    "MINIO_PASSWORD",
}


def setup_clearml(env_file_path: Union[str, Path, None] = None) -> None:
    """Set the environment variables required for ClearML to work.

    For it to work, ClearML needs to define the path of its configuration file as an environment
    variable, but this can be done internally as ClearML-Utils comes with a configuration file.
    However, other environment variables must be set, and this can be done with a `.env` file.

    Args:
        env_file_path: Path of the `.env` file defining environment variables; note that the
            `.env` file will not overwrite existing environment variables.
    """
    clearml_config_file_path: Path = _ROOT / "spider" / "clearml_utils" / "clearml.conf"
    os.environ["CLEARML_CONFIG_FILE"] = str(clearml_config_file_path)

    if env_file_path is not None:
        load_dotenv(dotenv_path=env_file_path)

    # Check that all environment values are set
    missing_var_names: set[str] = CLEARML_VAR_NAMES.difference(os.environ.keys())
    if missing_var_names:  # pragma: no cover
        warnings.warn(f"The following environment variables are not set: {missing_var_names}")

    # Check that all environment values are not empty
    non_empty_var_names: list[str] = [var_name for var_name in os.environ if os.environ[var_name]]
    empty_var_names: set[str] = CLEARML_VAR_NAMES.difference(non_empty_var_names)
    if empty_var_names:  # pragma: no cover
        warnings.warn(f"The following environment variables are empty: {empty_var_names}")

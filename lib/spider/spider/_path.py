from pathlib import Path

_ROOT = Path(__file__).parent.parent

ENV_FILE_PATH = _ROOT / ".env"
DATASET_PATH = _ROOT / "data" / "spider"
CONFIG_PATH = _ROOT / "configs"

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIGS_DIR = PROJECT_ROOT / "configs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

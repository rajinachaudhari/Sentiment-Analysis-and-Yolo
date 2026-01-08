from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DATA_YAML = DATA_DIR / "data.yaml"

MODEL_NAME = "yolov8n.pt"

IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 50
CONF_THRESHOLD = 0.4

DEVICE = "cuda"  # or "cpu"

RESULTS_DIR = BASE_DIR / "results"

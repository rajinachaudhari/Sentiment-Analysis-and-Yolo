from ultralytics import YOLO
from src.config import *

def evaluate_model():
    model = YOLO(RESULTS_DIR / "train/weights/best.pt")

    metrics = model.val(
        data=str(DATA_YAML),
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE
    )

    return metrics.results_dict

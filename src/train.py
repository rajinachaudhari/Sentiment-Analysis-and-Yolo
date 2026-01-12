from ultralytics import YOLO
from src.config import *

def train_model():
    model = YOLO(MODEL_NAME)

    model.train(
        data=str(DATA_YAML),
        epochs=10,
        imgsz=416,
        batch=2,
        device="cpu",
        project=str(RESULTS_DIR),
        name="train",
        exist_ok=True
    )
    
#     model.train(
#     data=str(DATA_YAML),
#     epochs=10,
#     imgsz=416,
#     batch=2,
#     device="cpu",
#     project=str(RESULTS_DIR),
#     name="train",
#     exist_ok=True
# )


if __name__ == "__main__":
    train_model()

from ultralytics import YOLO
from src.config import MODEL_NAME

def get_model(pretrained=True):
    if pretrained:
        return YOLO(MODEL_NAME)
    return YOLO("yolov8n.yaml")

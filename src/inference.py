import cv2
from ultralytics import YOLO
from src.config import *

def run_inference(image_path):
    model = YOLO(RESULTS_DIR / "train/weights/best.pt")

    results = model(image_path, conf=CONF_THRESHOLD)

    for r in results:
        img = r.plot()
        cv2.imshow("License Plate Detection", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

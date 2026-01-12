# from fastapi import FastAPI
# from pydantic import BaseModel
# import os
# import joblib
# import numpy as np
# import tensorflow as tf

# from src.preprocessing import clean_text


# app = FastAPI(title="Yolo API")

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# vectorizer = joblib.load("models_saved/tfidf_vectorizer.pkl")
# nb_model = joblib.load("models_saved/naive_bayes.pkl")
# dl_model = tf.keras.models.load_model("models_saved/deep_learning.h5")

# label_map = {0:"negative",1:"neutral",2:"positive"}

# class TextInput(BaseModel):
#     text: str
#     model: str = "naive_bayes"  

# @app.get("/")
# def health_check():
#     return {"status": "API running"}

# @app.post("/predict")
# def predict_sentiment(data: TextInput):
#     cleaned = clean_text(data.text)
#     vec = vectorizer.transform([cleaned])

#     if data.model == "deep_learning":
#         pred = dl_model.predict(vec.toarray())
#         sentiment = label_map[int(np.argmax(pred))]
#     else:
#         pred = nb_model.predict(vec)
#         sentiment = label_map[int(pred[0])]

#     return {
#         "input_text": data.text,
#         "model_used": data.model,
#         "predicted_sentiment": sentiment
#     }


from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
from pathlib import Path

from src.config import RESULTS_DIR, MODEL_NAME, CONF_THRESHOLD

app = FastAPI(title="YOLO License Plate Detection API")

# Resolve model path: prefer trained weights, otherwise fall back to the base model
trained_weights = RESULTS_DIR / "train/weights/best.pt"
model_path = trained_weights if trained_weights.exists() else Path(MODEL_NAME)
if not model_path.exists():
    raise RuntimeError("No model weights found. Train the model or place weights at results/train/weights/best.pt")

model = YOLO(str(model_path))


@app.get("/")
def health_check():
    return {"status": "API running", "model": str(model_path)}

@app.post("/predict")
async def predict(
    image: UploadFile = File(..., description="Upload an image"),
    conf: float = Query(0.4, ge=0.0, le=1.0),
):
    if not image.content_type.startswith("image"):
        raise HTTPException(status_code=400, detail="Please upload an image")

    image_bytes = await image.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    results = model(img, conf=conf)

    detections = []
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            detections.append({
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "confidence": float(b.conf[0]),
                "class_id": int(b.cls[0]),
                "label": r.names[int(b.cls[0])]
            })

    return {
        "num_detections": len(detections),
        "detections": detections
    }

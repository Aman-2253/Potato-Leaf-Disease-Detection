
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
import numpy as np
import io

# Create FastAPI application
app = FastAPI(title="Potato Disease Detection API")

# Load trained CNN model
MODEL_PATH = Path(__file__).resolve().parent / "model" / "potato_cnn_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

# IMPORTANT: Verify order against your training notebook
CLASS_NAMES = [
    "Early Blight",
    "Healthy",
    "Late Blight"
]

@app.get("/")
def home():
    return {"message": "Potato Disease API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image"
        )

    # Read uploaded image
    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    # Resize according to CNN model input
    height, width = model.input_shape[1:3]

    if height is None or width is None:
        raise HTTPException(
            status_code=500,
            detail="Model image dimensions are not fixed"
        )

    image = image.resize((width, height))

    # Convert image into NumPy array
    image_array = np.array(image, dtype=np.float32)

    # IMPORTANT:
    # Divide by 255 only if training used external
    # pixel normalization. Do not divide if the
    # model already contains a Rescaling layer.
    # image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(image_array, verbose=0)[0]

    if len(predictions) != len(CLASS_NAMES):
        raise HTTPException(
            status_code=500,
            detail="Model output does not match class names"
        )

    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index]) * 100

    all_scores = {
        name: round(float(score) * 100, 2)
        for name, score in zip(CLASS_NAMES, predictions)
    }

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": round(confidence, 2),
        "all_scores": all_scores
    }

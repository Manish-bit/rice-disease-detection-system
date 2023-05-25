import imghdr
from fastapi import FastAPI, File, HTTPException, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from typing import Tuple
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

origins = [
    "http://192.168.221.199",
    "http://192.168.221.199:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# script_directory = os.path.dirname(os.path.abspath(__file__))

# MODEL_PATH = os.path.join(script_directory, "..", "final models", "outlierrice")
# DISEASE_MODEL_PATH = os.path.join(script_directory, "..", "final models", "final2")


# MODEL = tf.keras.models.load_model(MODEL_PATH)
MODEL = tf.keras.models.load_model("https://github.com/Manish-bit/Models/tree/master/outlierrice")
CLASS_NAMES = ['No Rice', 'Rice']

# DISEASE_MODEL = tf.keras.models.load_model(DISEASE_MODEL_PATH)
DISEASE_MODEL = tf.keras.models.load_model("https://github.com/Manish-bit/Models/tree/master/final2")
DISEASE_CLASS_NAMES = ['Brown Spot', 'Healthy', 'Leaf Blight', 'Tungro']


def read_file_as_image(data: bytes, size: Tuple[int, int]) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.convert("RGB")
    image = image.resize(size)
    image = np.array(image)
    return image


def classify_image(model, image):
    image_batch = np.expand_dims(image, 0)
    predictions = model.predict(image_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    my_class = np.argmax(predictions[0])
    confidence = int(np.max(predictions[0])*100)
    print(predicted_class, confidence)
    return my_class, confidence


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    # Read the image
    image_data = await file.read()
    image1 = read_file_as_image(image_data, size=(256, 256))

    try:
        # Classify the image as a leaf or non-leaf image using binary classifier model
        is_leaf, confidence = classify_image(MODEL, image1)
        print(is_leaf)
        if is_leaf == 0:  # If not a leaf image, raise an HTTPException
            raise HTTPException(status_code=422, detail="Provided file is not a leaf image")
    except HTTPException as e:
        return {"error": str(e.detail)}
    

    # Predict the disease using your custom model
    image2 = read_file_as_image(image_data, size=(512, 512))
    img_batch = np.expand_dims(image2, 0)
    prediction = DISEASE_MODEL.predict(img_batch)
    predicted_class = DISEASE_CLASS_NAMES[np.argmax(prediction[0])]
    confidence = int(np.max(prediction[0])*100)
    print(predicted_class, "with a confidence of ", int(confidence*100))
    return {
        "predicted_class": predicted_class,
        "confidence": confidence
    }



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

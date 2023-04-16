from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from typing import Tuple
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../final models/final")
CLASS_NAMES = ['Brown Spot', 'Leaf Blast', 'Leaf Blight', 'Smut']



@app.get("/")
async def root():
    print("Hello world")
    return "Hello world"

# def read_file_as_image(data)->np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image

def read_file_as_image(data: bytes, size: Tuple[int, int]) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.resize(size)
    image = np.array(image)
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
   image = read_file_as_image(await file.read(), size=(512, 512))
   img_batch = np.expand_dims(image, 0)
   print(img_batch)
   prediction = MODEL.predict(img_batch)
   print(prediction)
   predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
   confidence = int(np.max(prediction[0])*100)
   print(predicted_class,"with a confidence of " ,int(confidence*100))
   return {
       "predicted_class":predicted_class,
       "confidence":confidence
   }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=5500)



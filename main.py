import io
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from model_loader import load_model
from predict import predict_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    print("Loading model...")
    app.state.model = load_model()
    print("Model loaded successfully.")


@app.post("/detect")
async def predict_api(image: UploadFile = File(...)):

    try:
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img)

        result = predict_image(app.state.model, img_array)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "API is running"}
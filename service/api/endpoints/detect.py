from fastapi import APIRouter, UploadFile, HTTPException
from PIL import Image
from fastapi.responses import JSONResponse
from io import BytesIO
import numpy as np
import cv2
from fastapi.responses import FileResponse
from service.core.logic.model_inference import determine_poverty
food_router = APIRouter()

@food_router.post("/detect")
async def detect(im: UploadFile):
    if im.filename.split(".")[-1] in ("jpg","jpeg","png"):
        pass
    else:
        raise HTTPException(status_codes = 415, detail = "Not an image")

    image = Image.open(BytesIO(im.file.read()))
    image = np.array(image)
    return determine_poverty(image)


from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
from ultralytics  import YOLO
import base64


model = YOLO('best.pt')

app = FastAPI()

def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file))
    return image

def encode_image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

@app.post("/infer/binary")
async def infer_binary(image: UploadFile = File(...)):
    image_data = await image.read()
    image = read_imagefile(image_data)
    
    prediction = model.predict(image)
    number_bags = len(prediction[0].boxes) 
    score = round(float(np.mean(prediction[0].boxes.cpu().numpy().conf)), 3)
    
    encoded_image = encode_image_to_base64(image)
    
    response = {
        "data": {
            "base64": encoded_image,
            "result": {
                "number_bags": number_bags,
                "score": score
            }
        },
        "msg": "success",
        "code": 200
    }
    return JSONResponse(content=response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

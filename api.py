from simple_ocr import recognize_plate
from time_series_model import forecast_time_series
from container_ocr import recognize_container_number

import os
from fastapi import FastAPI
from pydantic import BaseModel

images = os.listdir('images')
images = ['images/' + x for x in images]



cont_imgs = os.listdir('cont_imgs')
cont_imgs = ['cont_imgs/' + x for x in cont_imgs]


app = FastAPI()

class PredictRequest(BaseModel):
    mode: str                 # "ocr" или "ts"
    image_path: str | None = None
    n_future: int = 5

@app.post("/predict")
def predict(req: PredictRequest):
    if req.mode == "ocr":
        plate = recognize_plate(req.image_path)
        return {"mode": "ocr", "image_path": req.image_path, "plate": plate}

    elif req.mode == "ts":
        days, preds = forecast_time_series(req.n_future)
        return {"mode": "ts", "days": days, "predictions": preds}

    elif req.mode == "container":
        number = recognize_container_number(req.image_path)
        return {
            "mode": "container",
            "image_path": req.image_path,
            "container_number": number
        }

    else:
        return {"error": "mode must be 'ocr', 'ts', or 'container'"}



from fastapi.testclient import TestClient

client = TestClient(app)
'''
# Проверка OCR
resp_ocr = client.post("/predict", json={
    "mode": "ocr",
    "image_path": images[0]
})

print("OCR статус:", resp_ocr.status_code)
print("OCR ответ:", resp_ocr.json())



# Проверка временного ряда
resp_ts = client.post("/predict", json={
    "mode": "ts",
    "n_future": 5
})

print("TS статус:", resp_ts.status_code)
print("TS ответ:", resp_ts.json())
'''

# Проверка номеров контейнеров
for img in cont_imgs:  
    resp_container = client.post("/predict", json={
        "mode": "container",
        "image_path": img
    })
    print("Container:", resp_container.json())
    print('\n')
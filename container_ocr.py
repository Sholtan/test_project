# container_ocr.py

import os
import cv2
from ultralytics import YOLO

DETECTION_WEIGHTS = "detection.pt"
RECOGNITION_WEIGHTS = "recongition.pt"

def download_if_missing():
    if not os.path.exists(DETECTION_WEIGHTS):
        os.system(f"wget https://github.com/muratgokkaya/CNRS/raw/main/{DETECTION_WEIGHTS}")

    if not os.path.exists(RECOGNITION_WEIGHTS):
        os.system(f"wget https://github.com/muratgokkaya/CNRS/raw/main/{RECOGNITION_WEIGHTS}")

download_if_missing()

det_model = YOLO(DETECTION_WEIGHTS)
rec_model = YOLO(RECOGNITION_WEIGHTS)



def recognize_container_number(image_path: str) -> str | None:
    img = cv2.imread(image_path)
    if img is None:
        return None

    det_results = det_model(img, imgsz=640, conf=0.4, verbose=False)[0]
    detected_numbers = []

    for box in det_results.boxes:
        cls_idx = int(box.cls[0])
        cls_name = det_results.names[cls_idx]

        if cls_name not in ("container_number_h", "container_number_v"):
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        roi = img[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        rec_results = rec_model(roi, imgsz=640, conf=0.3, verbose=False)[0]

        char_boxes = []
        for cbox in rec_results.boxes:
            c_cls_idx = int(cbox.cls[0])
            char_label = rec_results.names[c_cls_idx]
            cx1, cy1, cx2, cy2 = map(int, cbox.xyxy[0].tolist())
            char_boxes.append((cx1, cy1, char_label))

        if not char_boxes:
            continue

        # Sort characters depending on orientation
        if cls_name == "container_number_h":
            char_boxes.sort(key=lambda t: t[0])  # left-to-right
        else:
            char_boxes.sort(key=lambda t: t[1])  # top-to-bottom

        recognized = "".join(char for _, _, char in char_boxes)
        detected_numbers.append(recognized)

    if not detected_numbers:
        return None

    return detected_numbers[0] 



from ultralytics import YOLO
import easyocr
import cv2

det_model = YOLO("license_plate_detector.pt")
ocr = easyocr.Reader(['en', 'ru'])

def recognize_plate(image_path: str) -> str:
    img = cv2.imread(image_path)
    results = det_model(img)[0]

    for box in results.boxes:
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            text = ocr.readtext(crop, detail=0)
            return " ".join(text)

    return "Номер не найден"

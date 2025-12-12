print('START\n\n')

from ultralytics import YOLO
import easyocr

import cv2

det_model = YOLO("license_plate_detector.pt")
ocr = easyocr.Reader(['en', 'ru'])   # , gpu=False

print('set gpu to false')

def recognize_plate(image_path: str) -> str:
    img = cv2.imread(image_path)
    results = det_model(img)[0]
    print('results are ready')
    for box in results.boxes:
        print('looping boxes')
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print("box coordinates:", x1, y1, x2, y2)
            crop = img[y1:y2, x1:x2]
            print('start ocr')
            text = ocr.readtext(crop, detail=0)
            print('reach here')
            return " ".join(text)

    return "Номер не найден"



img_path = 'images/1.jpg'
result = recognize_plate(img_path)
print(f"result: {result}")
print('DONE\n\n')




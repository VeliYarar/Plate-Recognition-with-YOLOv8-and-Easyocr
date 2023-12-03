import easyocr
from ultralytics import YOLO
import cv2

# EasyOCR modelini yükle
reader = easyocr.Reader(['en'],gpu=False)  # 'en' parametresi ingilizce dilini belirtir

model = YOLO("C:\\Users\\veli1\\Desktop\\projeler\\plaka_tanima\\gpu_plaka\\runs\\detect\\yolov8_plaka_detection\\weights\\best.pt")
cap = cv2.VideoCapture(0)

def metin_okuma(roi):
    result = reader.readtext(roi)
    
    for detection in result:
        print(detection[2])
        return detection[1]

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,(640,640))
    results = model.predict(frame)  # run prediction on img
    for result in results:  # iterate results
        boxes = result.boxes.cpu().numpy()  # get boxes on cpu in numpy
        for box in boxes:  # iterate boxes
            result.names[int(box.cls[0])]
            r = box.xyxy[0].astype(int)  # get corner points as int # print boxes
            x1, y1, x2, y2 = r[:4]
            roi = frame[y1:y2, x1+20:x2]
            sonuc = metin_okuma(roi)
            color = (0, 0, 255)
            cv2.putText(frame, sonuc, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, r[:2], r[2:], color, 2)

    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



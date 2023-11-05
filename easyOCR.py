import easyocr
from ultralytics import YOLO
import cv2
import requests
import json

# EasyOCR modelini yükle
reader = easyocr.Reader(['en'],gpu=False)  # 'en' parametresi ingilizce dilini belirtir

model = YOLO("C:\\Users\\veli1\\Desktop\\projeler\\plaka_tanima\\gpu_plaka\\runs\\detect\\yolov8_plaka_detection\\weights\\best.pt")
cap = cv2.VideoCapture(0)

def metin_okuma(roi):
    result = reader.readtext(roi)
    
    for detection in result:
        print(detection[2])
        return detection[1]

def request_fonk(text):
    if text is not None:  # Eğer sonuc değişkeni None değilse devam et
        text_without_spaces = text.replace(" ", "")
        payload = json.dumps({
        "plate": text_without_spaces,
        "key": "ZEEb8m>o8#QOPUvyI>HALdN,0r4eY6?Lm`+x/0,@ZSE#;:A.vf"
        })
        headers = {
        'Content-Type': 'application/json'
        }
        url = "https://platesystem.nwrobotic.com/api/Request/Control"
        try:
            response = requests.post(url, headers=headers, data=payload)
            data = response.json()  # JSON yanıtını doğrudan alın
            is_login_value = data.get("isLogin")  # Anahtarın varlığını kontrol etmek için get kullanın
            state = is_login_value
            return state
        except Exception as e:
            pass
    
    else:
        pass
global count
count = 0
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
            state = request_fonk(sonuc)
             
            if count == 0:
                color = (0, 0, 255)
            
            else:
                color = (0,255,0)

            if state == True:
                count = 1

            cv2.putText(frame, sonuc, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, r[:2], r[2:], color, 2)

    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



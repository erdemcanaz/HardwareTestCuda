from ultralytics import YOLO
from pathlib import Path
import cv2
import math 

script_path = Path(__file__).absolute()
script_folder = script_path.parent.absolute()

model_path = script_folder / "yolo_models" / "net_google_mask_28_06_2024.pt"
model = YOLO(model_path)  # pretrained YOLOv8n model
results = model(source=0, vid_stride=1, show=True, save=False, project=None, conf=0.5,stream = True)  # use webcam as source


# start webcam
desired_width = int(input("Enter the desired width of the frame: "))
desired_height = int(input("Enter the desired height of the frame: "))
cap = cv2.VideoCapture(0)
cap.set(3, desired_width)
cap.set(4, desired_height)

classNames = ["white_net", "blue_net", "safety_goggles", "blue_surgical_mask", "white_surgical_mask"]

while True:
    success, img = cap.read()
    results = model(img, show=True, save=False, project=None, conf=0.5,stream = True)

    # # coordinates
    # for r in results:
    #     boxes = r.boxes

    #     for box in boxes:
    #         # bounding box
    #         x1, y1, x2, y2 = box.xyxy[0]
    #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

    #         # put box in cam
    #         cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

    #         # confidence
    #         confidence = math.ceil((box.conf[0]*100))/100
    #         print("Confidence --->",confidence)

    #         # class name
    #         cls = int(box.cls[0])
    #         print("Class name -->", classNames[cls])

    #         # object details
    #         org = [x1, y1]
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         fontScale = 1
    #         color = (255, 0, 0)
    #         thickness = 2

    #         #cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    # cv2.imshow('Webcam', img)
    # if cv2.waitKey(1) == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()
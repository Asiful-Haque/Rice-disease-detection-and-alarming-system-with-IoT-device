from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2

model = YOLO("/home/sourab/Documents/rice detection/runs/detect/train/weights/best.pt")  # pretrained YOLOv8n model

# model.conf = 0.80
# model.iou = 0.10

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
      print("error in retrieving frame")
      break

    results = model([frame], stream=True)
    for result in results:
        for i, data in enumerate(result.boxes.data.tolist()):
            classs = result.boxes.cls.tolist()[i]

            confidence = data[4]
            print(confidence, result.names[classs])
            if confidence>0.50:
                xmin, ymin, xmax, ymax = (
                    int(data[0]),
                    int(data[1]),
                    int(data[2]),
                    int(data[3]),
                )
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # Using cv22.putText() method
                frame = cv2.putText(
                    frame,
                    f"{result.names[classs]}",
                    (xmin, ymin),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

    frame = cv2.resize(frame, (400, 400))

    cv2.imshow("ff", frame)
    # Press `q` to quit the program
    if cv2.waitKey(15) & 0xFF == ord('x'):
            break

# Release the webcam
cap.release()

cv2.destroyAllWindows()

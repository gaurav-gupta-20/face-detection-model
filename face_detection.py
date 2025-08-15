from ultralytics import YOLO 
import cv2

# Load trained YOLO model
model = YOLO("best.pt")  

# using laptop camera
cap = cv2.VideoCapture(0)# 0 means it uses ur default camera
# we can use phone camera also by downloading IP webcam on our phone or using usb
# To use our phone camera replace the above code with:-
# cap = cv2.VideoCapture("http://192.168.43.1:8080/video") replace ur ip address.


cv2.namedWindow("Real-Time Face Detection (Phone Camera)", cv2.WINDOW_NORMAL)# To resize the window


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run face detection
    results = model.predict(source=frame, conf=0.5, save=False)

    # Design Annotate frame
    annotated_frame = frame.copy()
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green
        conf = float(box.conf[0]) * 100
        label = f"Face: {conf:.1f}%"
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show output
    cv2.imshow("Real-Time Face Detection ", annotated_frame)

    # Exit with ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

from collections import defaultdict
import imutils
import pickle
import time
import cv2
import numpy as np
from ultralytics import YOLO

# === Student Information Mapping ===
student_info = {
    "Agnishwar Das": {
        "Name": "Agnishwar Das",
        "Roll": "14400121026",
        "Dept": "Computer Science",
        "Batch": "2021-25"
    },
    "Paramjeet Kumar Mahato": {
        "Name": "Paramjeet Kumar Mahato",
        "Roll": "14400121017",
        "Dept": "Computer Science",
        "Batch": "2021-25"
    },
    "Rifat Banu": {
        "Name": "Rifat Banu",
        "Roll": "14400121022",
        "Dept": "Computer Science",
        "Batch": "2021-25"
    },
    "Ritayan Sen": {
        "Name": "Ritayan Sen",
        "Roll": "14400121018",
        "Dept": "Computer Science",
        "Batch": "2021-25"
    },
    "Rohit Ghosh": {
        "Name": "Rohit Ghosh",
        "Roll": "14400121004",
        "Dept": "Computer Science",
        "Batch": "2021-25"
    },
    "Srijani Halder": {
        "Name": "Srijani Halder",
        "Roll": "14400121021",
        "Dept": "Computer Science",
        "Batch": "2021-25"
    },
    # Add more class labels and info here
}

# === Paths to model files and recognizer ===
embeddingModel = r"D:\PythonProject\Face_Recognition_DL\openface.nn4.small2.v1.t7"
recognizerFile = r"D:\PythonProject\Face_Recognition_DL\New_Output_Models\recognizer_Openface02.pickle"
labelEncFile = r"D:\PythonProject\Face_Recognition_DL\New_Output_Models\LE_Openface02.pickle"

# === Load YOLOv8 Face Detection Model ===
print("[INFO] Loading YOLOv8 face detector...")
yolo_model_path = r"D:\PythonProject\Face_Recognition_DL\model\yolov8n-face.pt"
yolo = YOLO(yolo_model_path)

# === Load Face Recognizer and Label Encoder ===
print("[INFO] Loading face recognizer and label encoder...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)
recognizer = pickle.loads(open(recognizerFile, "rb").read())
le = pickle.loads(open(labelEncFile, "rb").read())

# === Start Video Stream ===
print("[INFO] Starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(2.0)

recognized_names = []
detailed_mode = False  # Start in minimal display mode

while True:
    frame_counter = 0
    scores = defaultdict(float)

    while frame_counter < 100:
        ret, frame = cam.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=600)
        results = yolo.predict(frame, conf=0.4, verbose=False)
        boxes = results[0].boxes if results else []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = frame[y1:y2, x1:x2]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            # Get Face Embedding
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Predict Identity
            preds = recognizer.predict_proba([vec.flatten()])[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            if proba * 100 < 50:
                name = "Unknown"
            else:
                scores[name] += proba

            # Draw Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # Display Info Right of Bounding Box
            if name != "Unknown" and name in student_info:
                info = student_info[name]
                if detailed_mode:
                    text_lines = [
                        f"{info['Name']}",
                        f"Roll: {info['Roll']}",
                        f"Dept: {info['Dept']}",
                        f"Batch: {info['Batch']}",
                        f"Confidence: {proba * 100:.2f}%",
                    ]
                else:
                    text_lines = [
                        f"{info['Name']}",
                        f"Confidence: {proba * 100:.2f}%",
                    ]
            else:
                text_lines = [f"{name} : {proba * 100:.2f}%"]

            margin = 10
            font_scale = 0.7
            thickness = 2
            font_color = (255, 50, 50)  # Deep visible color (BGR)

            for idx, line in enumerate(text_lines):
                y = y1 + idx * 25
                x = x2 + margin
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                # Background rectangle for better visibility
                cv2.rectangle(frame, (x - 2, y - th), (x + tw + 2, y + 5), (30, 30, 30), -1)  # dark gray bg
                cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)

        cv2.imshow("Frame", frame)
        frame_counter += 1

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to break
            break
        elif key == ord("d"):  # Toggle detailed/minimal display
            detailed_mode = not detailed_mode
            print("[INFO] Display mode:", "Detailed" if detailed_mode else "Minimal")

    if key == 27:
        break

    if scores:
        name = max(scores.items(), key=lambda x: x[1])[0]
        print(f"[INFO] Attendance marked for: {name}")
        recognized_names.append(name)
    else:
        print("[INFO] No valid recognition for this set.")

    print("Press 'c' to continue recognizing another person, 'd' to toggle display mode, or 'q' to quit and generate report.")
    action = cv2.waitKey(0) & 0xFF

    if action == ord("q"):
        break
    elif action == ord("c"):
        print("[INFO] Continuing for next person...")
    elif action == ord("d"):
        detailed_mode = not detailed_mode
        print("[INFO] Display mode toggled to:", "Detailed" if detailed_mode else "Minimal")

# === Cleanup and Save Attendance ===
cam.release()
cv2.destroyAllWindows()

# Save attendance to text file
with open("recognized_names.txt", "w") as f:
    for name in recognized_names:
        if name in student_info:
            info = student_info[name]
            f.write(f"{info['Name']}, {info['Roll']}, {info['Dept']}, {info['Batch']}\n")
        else:
            f.write(f"{name}, Unknown Info\n")

print("[INFO] Attendance saved to recognized_names.txt")

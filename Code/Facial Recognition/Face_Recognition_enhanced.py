import cv2
import torch
import numpy as np
import imutils
import pickle
import time
from gfpgan import GFPGANer

# === Paths ===
embeddingModel = r"D:\PythonProject\Face_Recognition_DL\openface.nn4.small2.v1.t7"
recognizerFile = r"D:\PythonProject\Face_Recognition_DL\Output_Models\recognizer_Openface01.pickle"
labelEncFile = r"D:\PythonProject\Face_Recognition_DL\Output_Models\LE_Openface01.pickle"
prototxt_path = r"D:\PythonProject\Face_Recognition_DL\model\deploy.prototxt.txt"
model_path = r"D:\PythonProject\Face_Recognition_DL\model\res10_300x300_ssd_iter_140000.caffemodel"
gfpgan_model_path = r"D:\PythonProject\Face_Recognition_DL\model\GFPGANv1.4.pth"

# === Configuration ===
face_conf_threshold = 0.5
recognition_conf_threshold = 0.5

# === Load models ===
print("[INFO] Loading face detector...")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

print("[INFO] Loading face embedder...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

print("[INFO] Loading recognizer and label encoder...")
recognizer = pickle.loads(open(recognizerFile, "rb").read())
le = pickle.loads(open(labelEncFile, "rb").read())

print("[INFO] Loading GFPGAN model...")
gfpgan = GFPGANer(
    model_path=gfpgan_model_path,
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)

# === Start webcam ===
print("[INFO] Starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # Detect faces using DNN
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < face_conf_threshold:
            continue

        # Get face bounding box
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        face = frame[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]

        if fW < 20 or fH < 20:
            continue

        # === Enhance face using GFPGAN ===
        try:
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            _, restored_faces, _ = gfpgan.enhance(
                face_rgb, has_aligned=False, only_center_face=True, paste_back=False
            )
            enhanced_face = cv2.cvtColor(restored_faces[0], cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[WARNING] GFPGAN enhancement failed: {e}")
            enhanced_face = face

        # === Face recognition ===
        faceBlob = cv2.dnn.blobFromImage(enhanced_face, 1.0 / 255, (96, 96),
                                         (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(faceBlob)
        vec = embedder.forward()

        preds = recognizer.predict_proba([vec.flatten()])[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j] if proba >= recognition_conf_threshold else "Unknown"

        # === Draw results ===
        text = f"{name} : {proba * 100:.2f}%"
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 1)
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # Print recognition info
        if name != "Unknown":
            print(f"[INFO] Recognized: {name}, Probability: {proba * 100:.2f}%")
        else:
            print(f"[INFO] Unknown face detected with probability: {proba * 100:.2f}%")

    cv2.imshow("Frame", frame)

    if cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

# === Cleanup ===
cam.release()
cv2.destroyAllWindows()

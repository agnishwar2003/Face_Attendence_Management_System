# streamlit run Web.py --server.runOnSave false
import streamlit as st
import cv2
import numpy as np
import imutils
import pickle
from collections import defaultdict
from ultralytics import YOLO
import time
from datetime import datetime
import os
import pandas as pd

# ==== Load Models ====
embedding_model_path = r"D:\PythonProject\Face_Recognition_DL\openface.nn4.small2.v1.t7"
recognizer_path = r"D:\PythonProject\Face_Recognition_DL\New_Output_Models\recognizer_Openface02.pickle"
label_encoder_path = r"D:\PythonProject\Face_Recognition_DL\New_Output_Models\LE_Openface02.pickle"
yolo_model_path = r"D:\PythonProject\Face_Recognition_DL\model\yolov8n-face.pt"

embedder = cv2.dnn.readNetFromTorch(embedding_model_path)
recognizer = pickle.loads(open(recognizer_path, "rb").read())
label_encoder = pickle.loads(open(label_encoder_path, "rb").read())
yolo = YOLO(yolo_model_path)

# ==== Student Information ====
student_info = {
    "Agnishwar Das": {"Name": "Agnishwar Das", "Roll": "14400121026", "Dept": "Computer Science", "Batch": "2021-25"},
    "Paramjeet Kumar Mahato": {"Name": "Paramjeet Kumar Mahato", "Roll": "14400121017", "Dept": "Computer Science", "Batch": "2021-25"},
    "Rifat Banu": {"Name": "Rifat Banu", "Roll": "14400121022", "Dept": "Computer Science", "Batch": "2021-25"},
    "Ritayan Sen": {"Name": "Ritayan Sen", "Roll": "14400121018", "Dept": "Computer Science", "Batch": "2021-25"},
    "Rohit Ghosh": {"Name": "Rohit Ghosh", "Roll": "14400121004", "Dept": "Computer Science", "Batch": "2021-25"},
    "Srijani Halder": {"Name": "Srijani Halder", "Roll": "14400121021", "Dept": "Computer Science", "Batch": "2021-25"},
}

# ==== Page Layout Config ====
st.set_page_config(page_title="Face Recognition Attendance", layout="wide")

# ==== Sidebar Navigation ====
with st.sidebar:
    st.title("📂 Navigation")
    page = st.radio("Go to", ["Home", "Take Attendance", "View Attendance"])

# ==== Home Page ====
def home_page():
    st.markdown("<h2>🏠 Face Attendence System</h2>", unsafe_allow_html=True)
    st.write("""
    Welcome to the Neotia Institute of Technology, Management and Science.  
    Please use the **sidebar** to navigate to the **Take Attendance** page and mark your attendance by simply looking at the camera.  
    Your attendance will be recorded automatically with high accuracy and ease.  

    Stay safe and have a great day!
    """)

# ==== Take Attendance Page ====
def take_attendance_page():
    st.markdown("<h3>📸 Face Recognition: Take Attendance</h3>", unsafe_allow_html=True)
    if "run_video" not in st.session_state:
        st.session_state.run_video = False

    control_col, video_col, result_col = st.columns([1, 2, 1])

    with control_col:
        # st.markdown('<h4 style="font-size:18px;">🎛️ Control Panel</h4>', unsafe_allow_html=True)
        if st.button("📷 Start Camera"):
            st.session_state.run_video = True
        if st.button("🛑 Stop Camera"):
            st.session_state.run_video = False
        start_button = st.button("📸 Mark Attendence")

    frame_placeholder = video_col.empty()

    with result_col:
        st.subheader("📋 Results")
        result_placeholder = st.empty()
        final_class_placeholder = st.empty()
        time_placeholder = st.empty()
        student_info_placeholder = st.empty()
        face_preview_placeholder = st.empty()

    def video_stream():
        cap = cv2.VideoCapture(0)
        time.sleep(2.0)
        while st.session_state.run_video:
            ret, frame = cap.read()
            if not ret:
                st.warning("❌ Camera not accessible.")
                break
            frame = imutils.resize(frame, width=450)
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            time.sleep(0.03)
        cap.release()
        frame_placeholder.empty()

    if st.session_state.run_video:
        video_stream()

    if start_button:
        cap = cv2.VideoCapture(0)
        time.sleep(2.0)
        scores_accum = defaultdict(float)
        frame_count = 0
        last_face = None

        with control_col:
            st.info("Capturing frames...")

        while frame_count < 50:
            ret, frame = cap.read()
            if not ret:
                st.warning("❌ Camera not accessible.")
                break

            frame = imutils.resize(frame, width=450)
            results = yolo.predict(frame, conf=0.4, verbose=False)
            boxes = results[0].boxes if results else []

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = frame[y1:y2, x1:x2]
                last_face = face.copy()
                if face.shape[0] < 20 or face.shape[1] < 20:
                    continue

                face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(face_blob)
                vec = embedder.forward()

                preds = recognizer.predict_proba([vec.flatten()])[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = label_encoder.classes_[j]

                if proba * 100 >= 50:
                    scores_accum[name] += proba
                break

            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            if last_face is not None:
                small_face = cv2.resize(last_face, (250, 300))
                face_preview_placeholder.image(
                    cv2.cvtColor(small_face, cv2.COLOR_BGR2RGB),
                    caption="Detected Face",
                    use_container_width=False,
                    channels="RGB"
                )
            else:
                face_preview_placeholder.empty()

            frame_count += 1

        cap.release()

        if scores_accum:
            final_scores = {k: v / frame_count for k, v in scores_accum.items()}
            sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            top_name, top_score = sorted_scores[0]

            result_text = "<br>".join([f"🔹 <b>{name}</b>: {score * 100:.2f}%" for name, score in sorted_scores])
            result_placeholder.markdown(result_text, unsafe_allow_html=True)
            final_class_placeholder.markdown(f"✅ <b>Final Identity:</b> `{top_name}`", unsafe_allow_html=True)

            timestamp = datetime.now().strftime('%d %B %Y, %I:%M %p')
            time_placeholder.markdown(f"🕒 <b>Time:</b> `{timestamp}`", unsafe_allow_html=True)

            if top_name in student_info:
                info = student_info[top_name]
                student_info_placeholder.markdown(
                    f"""<div style="border:1px solid #ccc; border-radius:10px; padding:10px; background-color:#f9f9f9">
                    <b>👤 Name:</b> {info['Name']}<br>
                    <b>🆔 Roll:</b> {info['Roll']}<br>
                    <b>🏫 Dept:</b> {info['Dept']}<br>
                    <b>🎓 Batch:</b> {info['Batch']}
                    </div>""",
                    unsafe_allow_html=True
                )

                df_attendance = pd.DataFrame({
                    "Name": [info["Name"]],
                    "Roll": [info["Roll"]],
                    "Department": [info["Dept"]],
                    "Batch": [info["Batch"]],
                    "Confidence (%)": [round(top_score * 100, 2)],
                    "Timestamp": [timestamp]
                })

                attendance_file = "Attendance_Sheet.csv"
                if os.path.exists(attendance_file):
                    df_attendance.to_csv(attendance_file, mode='a', header=False, index=False)
                else:
                    df_attendance.to_csv(attendance_file, index=False)

                with control_col:
                    st.success(f"📁 Attendance marked in `{attendance_file}`")

                with control_col:
                    if os.path.exists(attendance_file):
                        with open(attendance_file, "rb") as file:
                            st.download_button(
                                label="📤 Download Attendance Sheet",
                                data=file,
                                file_name="Attendance_Sheet.csv",
                                mime="text/csv"
                            )
            else:
                student_info_placeholder.markdown("❓ No student information found.")
        else:
            result_placeholder.markdown("⚠️ No valid face recognized.")
            final_class_placeholder.markdown("❌ <b>Final Identity:</b> `None`", unsafe_allow_html=True)
            time_placeholder.empty()
            student_info_placeholder.empty()

# ==== View Attendance Page ====
def view_attendance_page():
    st.markdown("<h3>📊 Attendance Records</h3>", unsafe_allow_html=True)
    attendance_file = "Attendance_Sheet.csv"
    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
        st.dataframe(df, use_container_width=True)

        with st.expander("🔍 Filter"):
            name_filter = st.text_input("Search by Name:")
            date_filter = st.date_input("Filter by Date", value=None)

            if name_filter:
                df = df[df["Name"].str.contains(name_filter, case=False)]

            if date_filter:
                df = df[df["Timestamp"].str.startswith(str(date_filter))]

            st.dataframe(df)

        st.download_button("📥 Download CSV", data=df.to_csv(index=False), file_name="Filtered_Attendance.csv")
    else:
        st.warning("⚠️ No attendance records found yet.")

# ==== Render Page Based on Navigation ====
if page == "Home":
    home_page()
elif page == "Take Attendance":
    take_attendance_page()
elif page == "View Attendance":
    view_attendance_page()

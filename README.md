# Face Recognition Attendance System

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [System Architecture](#system-architecture)  
- [Requirements](#requirements)  
- [Setup Instructions](#setup-instructions)  
- [Usage](#usage)  
- [Code Highlights](#code-highlights)  
- [Results and Discussion](#results-and-discussion)  
- [Graphs and Visuals](#graphs-and-visuals)  
- [Future Work](#future-work)  
- [Acknowledgments](#acknowledgments)  
- [Contact](#contact)    

---

## Overview

This project implements a **real-time face recognition-based attendance system** leveraging state-of-the-art deep learning models and classical machine learning techniques. The system detects faces using both SSD and YOLOv8 detection models, enhances low-resolution faces using GFPGAN, generates embeddings via OpenFace, and recognizes individuals using an SVM classifier. It provides a user-friendly interface using Streamlit to capture attendance automatically from webcam feeds.

---

## Features

- Real-time face detection with **SSD (ResNet-based)** and **YOLOv8n-face** models.  
- Face enhancement for low-quality images using **GFPGAN**.  
- Face embeddings extracted by **OpenFace Torch model**.  
- Support Vector Machine (SVM) classifier for accurate face recognition.  
- Attendance logging with timestamps, recognition confidence, and student metadata.  
- Streamlit web app for interactive attendance taking and record viewing.  
- Downloadable and filterable attendance CSV files.  
- High accuracy and efficient real-time processing on CPU.

---

## System Architecture

1. **Face Detection:**  
   - OpenCV DNN with SSD Res10 model (Caffe) for accurate face detection at ~16.67 FPS.  
   - YOLOv8n-face model for efficient detection (~13 FPS on CPU).

2. **Face Enhancement:**  
   - GFPGAN model restores and enhances cropped faces for improved recognition.

3. **Face Embedding Extraction:**  
   - OpenFace model produces 128-dimensional embedding vectors from aligned faces.

4. **Face Recognition:**  
   - SVM classifier trained on embeddings to identify individuals with high precision.

5. **Attendance Module:**  
   - Automatically logs recognized individuals with detailed info and timestamps in CSV.

6. **User Interface:**  
   - Streamlit-based app for webcam control, attendance capture, result display, and CSV management.

---

## Requirements

- Python 3.7+  
- Packages:  
  - `opencv-python`  
  - `numpy`  
  - `imutils`  
  - `pickle`  
  - `ultralytics` (YOLOv8)  
  - `gfpgan`  
  - `streamlit`  
  - `pandas`  
  - `scikit-learn`  

Install dependencies using:

```bash
pip install opencv-python numpy imutils ultralytics gfpgan streamlit pandas scikit-learn
```

## Setup Instructions

1. **Clone the repository** to your local machine.

2. **Download the required models:**  
   - OpenFace embedding model (`openface.nn4.small2.v1.t7`)  
   - SSD face detection model:  
     - `deploy.prototxt.txt`  
     - `res10_300x300_ssd_iter_140000.caffemodel`  
   - YOLOv8 face detection weights (`yolov8n-face.pt`)  
   - GFPGAN pretrained weights (`GFPGANv1.4.pth`)

3. **Place model files** in directories according to paths specified in the code.

4. **Prepare dataset and train SVM classifier:**  
   - Extract embeddings using OpenFace on your face images.  
   - Train SVM classifier and save the model and label encoder as `.pickle` files.

5. **Run the Streamlit app:**

```bash
streamlit run Web.py
```

## Usage

### Home Page

- Introduction and usage instructions.

### Take Attendance

- Start the webcam, detect faces, recognize identities, and mark attendance automatically.  
- View real-time results and preview detected faces.  
- Download attendance CSV.

### View Attendance

- Load saved attendance records.  
- Filter by name or date.  
- Download filtered attendance CSV.

---

## Code Highlights

- **Face Detection:** SSD Caffe model and YOLOv8 (Ultralytics) model integration.  
- **Face Embeddings:** OpenFace Torch `.t7` model loaded with OpenCV DNN.  
- **Recognition:** SVM classifier with probability thresholding.  
- **Face Enhancement:** GFPGAN to improve recognition on low-resolution faces.  
- **Real-time Streaming:** Streamlit UI with start/stop camera controls, result display, and attendance export.

---

## Results and Discussion

### 5.1 Accuracy Metrics

#### 5.1.1 Face Detection Performance

**Face Detection using SSD and Caffe Framework:**

- Total frames processed: 245  
- Frames with face detected: 245 (100%)  
- Average confidence score: 0.97 (range 0.91 - 1.00)  
- Average processing speed: 16.67 FPS  

[INFO] Total Frames Processed: 245
[INFO] Frames with Face Detected: 245
[INFO] Final Detection Rate: 100.00%
[INFO] Average FPS: 16.67
[INFO] Avg Confidence of Detections: 0.97
[INFO] Confidence Range: Min=0.91, Max=1.00

![image](https://github.com/user-attachments/assets/74c00ad0-7b2c-410c-8077-2ec69fecb079)

_Figure : Face Detection (SSD)_

---

**Face Detection using YOLOv8:**

- Detection rate: 100%  
- Average confidence: 0.80 (range 0.42 - 0.86)  
- Average FPS: 13.02 (CPU)  

[INFO] Final Detection Rate: 100.00%
[INFO] Average FPS: 13.02
[INFO] Avg Confidence of Detections: 0.80
[INFO] Confidence Range: Min=0.42, Max=0.86

![image](https://github.com/user-attachments/assets/1e2cb9fc-44ed-4bc7-80d5-f4db6e7b49e9)

_Figure : Face Detection (YOLOv8)_

---

#### 5.1.2 Model Performance During Training

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.9983 |
| Precision | 0.9983 |
| Recall    | 0.9983 |

_Table 1: Performance During Training_

**Confusion Matrix:**

| Actual \ Predicted | 0   | 1   | 2   | 3   | 4   | 5   |
|--------------------|-----|-----|-----|-----|-----|-----|
| 0                  | 500 | 0   | 0   | 0   | 0   | 0   |
| 1                  | 0   | 500 | 0   | 0   | 0   | 0   |
| 2                  | 0   | 2   | 496 | 0   | 0   | 0   |
| 3                  | 0   | 0   | 1   | 499 | 0   | 0   |
| 4                  | 0   | 0   | 0   | 0   | 500 | 0   |
| 5                  | 0   | 0   | 0   | 0   | 0   | 500 |

_Table 2: Confusion Matrix_

**Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 1.00      | 1.00   | 1.00     | 500     |
| 1     | 0.99      | 1.00   | 1.00     | 500     |
| 2     | 1.00      | 1.00   | 1.00     | 500     |
| 3     | 1.00      | 0.99   | 1.00     | 500     |
| 4     | 1.00      | 1.00   | 1.00     | 500     |
| 5     | 1.00      | 1.00   | 1.00     | 500     |

_Table 3: Classification Report_

**Overall Metrics:**  
- Accuracy: 1.00  
- Macro Average: Precision = 1.00, Recall = 1.00, F1-Score = 1.00  
- Weighted Average: Precision = 1.00, Recall = 1.00, F1-Score = 1.00  

![roc_curve_Openface10](https://github.com/user-attachments/assets/0b9a3adf-6171-4295-b096-bbcf1dc6e87d)

_Figure : ROC Curve of Training Performance_

---

#### 5.1.3 Face Recognition Performance

| Metric                          | Value  |
|--------------------------------|--------|
| Total Frames Processed          | 144    |
| Total Faces Detected            | 144    |
| Recognized Faces                | 124    |
| Unknown Faces                  | 20     |
| Recognition Rate                | 86.11% |
| Average Confidence (Recognized) | 0.92   |

_Table 4: Performance During Recognition_

- Detection rate was 100% across all frames.  
- Recognition accuracy was 86.11% with high confidence of 92%.  
- The remaining 13.89% were classified as unknown due to confidence thresholds.

---

## 5.1.3 Graphs and Visuals

### Facial Detection and Dataset Creation Phase

![03  Screenshot (387)](https://github.com/user-attachments/assets/f62ded7a-e271-47e7-ad45-c88ade82ae34)

_Figure : Real Time Dataset Creation_

![02  Screenshot 2025-06-07 094219](https://github.com/user-attachments/assets/dcc86f5e-0da8-47d8-8492-a968233a944b)

_Figure : Sample Image Dataset_

### Facial Embedding Generation 

![embedding_visualization](https://github.com/user-attachments/assets/8e2ca03b-8ce0-4e3d-b2ed-1e14070ed95a)

_Figure : Visualization of Each classes Embedding_

### Recognition and Information Visualization

![04  Screenshot (390)](https://github.com/user-attachments/assets/088aedf8-1122-417f-8207-3447aa74b225)

_Figure : Visualizing Class_Label along with Student Information_

### Web User Interface

![05  Screenshot (396)](https://github.com/user-attachments/assets/5d23970a-f25a-42b6-86ed-af55b3ce6e33)

_Figure : Home page_

![06  Screenshot (397)](https://github.com/user-attachments/assets/7056e715-1cc5-43f2-9974-f2124038cb57)

_Figure : Face Scan and Attendance Marking Page_

![07  Screenshot (399)](https://github.com/user-attachments/assets/a8cf64bb-52ab-403b-baf6-79e784b332e8)

_Figure : View Attendance Page_

---

## 4.10 Challenges Encountered

During the development and deployment of the face recognition attendance system, several challenges were faced, stemming from both technological limitations and environmental/hardware constraints:

1. **False Positives**  
   The system occasionally misidentified non-face regions (e.g., posters, shadows) as faces, especially with YOLOv8 detection. These false detections sometimes received moderate confidence scores from the SVM classifier, leading to incorrect attendance entries.

2. **Low-Light Performance**  
   Poor lighting conditions degraded detection and recognition accuracy. YOLOv8 showed inconsistent face detection with missed or fluctuating bounding boxes, while OpenFace embeddings had lower confidence, causing misclassifications in dim or unevenly lit environments.

3. **Similar Face Issues**  
   Individuals with visually similar facial features produced overlapping embeddings in feature space, causing confusion in the classifier. This led to occasional misclassifications and incorrect attendance markings for such visually similar faces.

4. **Hardware Limitations**  
   Running computationally intensive models like YOLOv8 and OpenFace embedding extraction on CPU without GPU acceleration resulted in slow inference speeds, with frame rates often dropping below 10 FPS. This impaired real-time performance and interface responsiveness.

---

## Future Work

- Implement GPU acceleration for faster real-time processing.  
- Increase dataset size and diversity for better generalization.  
- Support multi-face attendance in crowded scenes.  
- Integrate with institutional attendance management systems.

---

## Acknowledgments

- OpenFace: [https://cmusatyalab.github.io/openface/](https://cmusatyalab.github.io/openface/)  
- Ultralytics YOLOv8: [https://ultralytics.com/](https://ultralytics.com/)  
- GFPGAN: [https://github.com/TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN)  

---

# Contact

**Agnishwar Das**  
Email: your-email@example.com  
GitHub: [https://github.com/yourusername](https://github.com/yourusername)  

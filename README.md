# 👁️‍🗨️ Face Recognition Attendance System using OpenFace, OpenCV & SVM:

This project implements a real-time face recognition-based class attendance system. It uses deep learning-based face detection (SSD with ResNet backbone), OpenFace for 128D face embeddings, and a Support Vector Machine (SVM) classifier for identity recognition. The system captures live video through a webcam, identifies the student, and logs their attendance with a timestamp in a CSV file.

📂 Table of Contents

1. Code Architecture

2. Data Collection

3. Embedding Generation

4. Model Training & Evaluation

5. Recognition Process

6. Challenges Encountered

7. Sample Results

8. How to Run

9. Requirements

10. Future Improvements

## ✅ Code Architecture
The system is modularized into five key scripts, each with a clear responsibility:

### ssd.ipynb

a. Uses a pre-trained SSD face detector (ResNet-based) to detect faces from webcam feed.

b. Captures 100 grayscale face images per user.

c. Crops , resize(here 96*96 pixels) and saves these images to a structured folder (dataset/<username>/).

### face_embedding_Openface.py

a. Loads all collected face images and performs face alignment to normalize pose and scale.

b. Feeds aligned faces to the OpenFace model.

c. Extracts a 128-dimensional embedding vector per image.

d. Saves embeddings and corresponding labels in a serialized format (embeddings.pkl).

### face_train_SVM.py

a. Loads the saved embeddings and associated labels.

b. Encodes textual labels into numerical values.

c. Trains a linear Support Vector Machine (SVM) classifier.

d. Evaluates the model using classification metrics and visualizations.

e. Saves the trained SVM model and label encoder for inference.

### FaceRecognition_Attendence_YOLO.py

a. Activates the webcam and detects faces in real-time.

b. Aligns detected faces and extracts embeddings using the OpenFace model.

c. Uses the trained SVM model to predict identity.

d. Collects predictions over 100 frames and selects the most frequent label via majority voting.

e. Logs the recognized label along with the current date and time using the logging module.

### attendance_final.py
a. Handles the logging of attendance data.

b. Avoids duplicate entries for the same user on the same date.

c. Stores attendance records in attendance.csv in the format: Name, Date, Time.


## 🎥 Data Collection

### Webcam Integration

The webcam is used as the primary device for capturing training data. Users are prompted to enter their name, after which the system captures a sequence of face images in real time.

### Number of Images Per User

Each user provides 100 grayscale face images to account for variations in lighting, expression, and minor pose changes. System will capture total 500 images under 5 batches in realtime, means 100 images for each batch. This helps generalize the model and reduce overfitting.

### Storage Format
Images are stored in a directory-based structure:

dataset/

├── User1/

│   ├── 1.jpg

│   ├── 2.jpg

│   └── ...

├── User2/

│   ├── 1.jpg

│   ├── 2.jpg

│   └── ...

└── ...



All images are saved in grayscale to reduce noise and computational complexity.

## 🧠 Embedding Generation

### Face Alignment

Before passing images to the embedding model, facial landmarks are used to align the face. This ensures that eyes, nose, and mouth are consistently positioned across all samples, which improves embedding quality.

### Vector Creation

Each aligned face is passed through the OpenFace model, which generates a 128-dimensional embedding vector that uniquely represents the identity features of the face.

### Embedding Storage

Embeddings are saved in a serialized .pkl file along with their corresponding labels. This file serves as the input for the model training phase.


## 🧪 Model Training & Evaluation

### 📊 Dataset Split

The dataset comprises 3000 facial embeddings derived from six individuals. These embeddings are split into training (80%) and testing (20%) using stratified sampling, ensuring each individual is proportionally represented in both sets.

### 📈 Training Metrics

After training the classifier (SVM), the evaluation results on the test set are:

-> Total Embeddings: 3000

-> Individuals Recognized: 'Paramjeet Kumar Mahato', 'Ritayan Sen', 'Rifat Banu', 'Srijani Halder', 'Agnishwar_Das', 'Rohit Ghosh'

### 🔍 Performance Scores:

-> Accuracy: 0.9983

-> Precision: 0.9983

-> Recall: 0.9983

These high metrics indicate the classifier generalizes extremely well to unseen data.

### 📊 Confusion Matrix

|              | Predicted 0 | Predicted 1 | Predicted 2 | Predicted 3 | Predicted 4 | Predicted 5 |
| ------------ | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| **Actual 0** | 500         | 0           | 0           | 0           | 0           | 0           |
| **Actual 1** | 0           | 500         | 0           | 0           | 0           | 0           |
| **Actual 2** | 0           | 2           | 496         | 0           | 0           | 0           |
| **Actual 3** | 0           | 0           | 1           | 499         | 0           | 0           |
| **Actual 4** | 0           | 0           | 0           | 0           | 500         | 0           |
| **Actual 5** | 0           | 0           | 0           | 0           | 0           | 500         |


 This matrix shows almost perfect classification across all six classes, with only a few misclassifications between classes 1–2 and 2–3, which might result from facial similarities or variations in lighting.

### 🧾 Classification Report

| **Class** | **Precision** | **Recall** | **F1-Score** | **Support** |
| :-------: | :-----------: | :--------: | :----------: | :---------: |
|   **0**   |      1.00     |    1.00    |     1.00     |     500     |
|   **1**   |      0.99     |    1.00    |     1.00     |     500     |
|   **2**   |      1.00     |    1.00    |     1.00     |     500     |
|   **3**   |      1.00     |    0.99    |     1.00     |     500     |
|   **4**   |      1.00     |    1.00    |     1.00     |     500     |
|   **5**   |      1.00     |    1.00    |     1.00     |     500     |

Overall Metrics:

a. Accuracy: 1.00

b. Macro Avg: Precision: 1.00, Recall: 1.00, F1-Score: 1.00

c. Weighted Avg: Precision: 1.00, Recall: 1.00, F1-Score: 1.00

These results confirm that the face recognition classifier achieves excellent performance with near-zero error rates.

### 📉 ROC Analysis

Given the multiclass nature of the problem, a One-vs-Rest ROC analysis was performed for each individual class. All ROC curves showed Area Under Curve (AUC) values close to 1.0, confirming high separability among classes.

![roc_curve_Openface07](https://github.com/user-attachments/assets/b757e68f-1d1a-427f-9cb0-71f1e5dd4a79)



## 🧾 Recognition Process

### Real-Time Face Detection

The system activates the webcam and continuously captures video frames. For each frame, it detects whether a face is present and extracts it.

### Embedding Comparison

Once a face is detected, it is aligned and converted into a 128D embedding. This embedding is then passed to the trained SVM model to obtain a predicted label.

### Attendance Logging

Predictions are collected over 100 frames, and a majority vote is used to finalize the identity. The recognized label is then recorded in the attendance CSV file along with the current timestamp.





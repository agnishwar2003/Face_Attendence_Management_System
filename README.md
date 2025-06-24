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

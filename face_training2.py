import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Define paths
embeddingFile = r"D:\PythonProject\Face_Recognition_DL\output\embeddings_Openface_Agnishwar.pickle"
recognizerFile = r"D:\PythonProject\Face_Recognition_DL\output\recognizer_Openface_Agnishwar.pickle"
labelEncFile = r"D:\PythonProject\Face_Recognition_DL\output\le_Openface_Agnishwar.pickle"

# Load embeddings
print("[INFO] Loading face embeddings...")
with open(embeddingFile, "rb") as f:
    data = pickle.load(f)

# Check embeddings data
if "names" not in data or "embeddings" not in data:
    raise ValueError("[ERROR] Invalid embeddings file format. Ensure the embeddings were generated correctly.")

print(f"[INFO] Total embeddings: {len(data['embeddings'])}")
print(f"[INFO] Unique names found: {set(data['names'])}")

# Encode labels
print("[INFO] Encoding labels...")
labelEnc = LabelEncoder()
labels = labelEnc.fit_transform(data["names"])

# Check number of unique labels
unique_labels = set(labels)
if len(unique_labels) <= 1:
    print("[ERROR] Not enough classes to train. Please add more people to the dataset.")
    exit()

# Train the model
print("[INFO] Training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# Ensure output directory exists
os.makedirs(os.path.dirname(recognizerFile), exist_ok=True)

# Save trained model
with open(recognizerFile, "wb") as f:
    pickle.dump(recognizer, f)
with open(labelEncFile, "wb") as f:
    pickle.dump(labelEnc, f)

print("[INFO] Recognizer and label encoder saved successfully.")

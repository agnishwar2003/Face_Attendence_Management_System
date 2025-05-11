import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# Define paths
embeddingFile = r"D:\PythonProject\Face_Recognition_DL\Output_Models\embeddings_Openface08.pickle"
recognizerFile = r"D:\PythonProject\Face_Recognition_DL\Output_Models\recognizer_Openface06.pickle"
labelEncFile = r"D:\PythonProject\Face_Recognition_DL\Output_Models\LE_Openface06.pickle"

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

# Make predictions
predictions = recognizer.predict(data["embeddings"])

# Calculate accuracy, precision, recall, confusion matrix
accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions, average='weighted')
recall = recall_score(labels, predictions, average='weighted')
conf_matrix = confusion_matrix(labels, predictions)
class_report = classification_report(labels, predictions)

# Print metrics
print(f"[INFO] Accuracy: {accuracy:.4f}")
print(f"[INFO] Precision: {precision:.4f}")
print(f"[INFO] Recall: {recall:.4f}")
print(f"[INFO] Confusion Matrix:\n{conf_matrix}")
print(f"[INFO] Classification Report:\n{class_report}")

# Ensure output directory exists
os.makedirs(os.path.dirname(recognizerFile), exist_ok=True)

# Save trained model
with open(recognizerFile, "wb") as f:
    pickle.dump(recognizer, f)
with open(labelEncFile, "wb") as f:
    pickle.dump(labelEnc, f)

print("[INFO] Recognizer and label encoder saved successfully.")

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import os

# Initializing the embedding and recognizer file paths
embeddingFile = r"D:\PythonProject\Face_Recognition_DL\output\embeddings_Openface_Agnishwar.pickle"  # Updated path for your embeddings
recognizerFile = "output/recognizer_Openface_Agnishwar.pickle"
labelEncFile = "output/le_Openface_Agnishwar.pickle"

print("Loading face embeddings...")
with open(embeddingFile, "rb") as f:
    data = pickle.load(f)

print("Encoding labels...")
labelEnc = LabelEncoder()
labels = labelEnc.fit_transform(data["names"])

# Check the unique labels
unique_labels = set(labels)
if len(unique_labels) <= 1:
    raise ValueError(f"The number of classes has to be greater than one; got {len(unique_labels)} class(es)")

print(f"Unique labels: {unique_labels}")

print("Training model")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# Ensure the output directory exists
output_dir = os.path.dirname(recognizerFile)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the recognizer model
with open(recognizerFile, "wb") as f:
    pickle.dump(recognizer, f)

# Save the label encoder
with open(labelEncFile, "wb") as f:
    pickle.dump(labelEnc, f)

print("Recognizer and label encoder saved successfully.")

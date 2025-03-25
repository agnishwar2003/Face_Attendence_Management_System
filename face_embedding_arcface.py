import os
import pickle
import numpy as np
import cv2
from imutils import paths
from deepface import DeepFace

# Paths to your dataset and output files
dataset = r"D:\PythonProject\Face_Recognition_DL\Face_recognition_Dataset"
embeddingFile = "output/embeddings_arcface.pickle"

def get_embedding(face_img):
    try:
        embedding = DeepFace.represent(face_img, model_name='ArcFace', detector_backend='retinaface')
        return embedding[0]['embedding'] if embedding else None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Get image paths
imagePaths = list(paths.list_images(dataset))

# Initialize lists for known embeddings and names
knownEmbeddings = []
knownNames = []
total = 0

# Process each image
for (i, imagePath) in enumerate(imagePaths):
    print(f"Processing image {i + 1}/{len(imagePaths)}")
    
    # Get the folder name (which represents the person's name)
    name = os.path.basename(os.path.dirname(imagePath))
    
    # Read image
    image = cv2.imread(imagePath)
    if image is None:
        print(f"Skipping invalid image: {imagePath}")
        continue
    
    # Get ArcFace embedding
    embedding = get_embedding(image)
    if embedding is not None:
        knownNames.append(name)
        knownEmbeddings.append(embedding)
        total += 1

print(f"Total Embeddings: {total}")

# Save embeddings to a file
data = {"embeddings": knownEmbeddings, "names": knownNames}
output_dir = os.path.dirname(embeddingFile)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(embeddingFile, "wb") as f:
    pickle.dump(data, f)

print("Process Completed")

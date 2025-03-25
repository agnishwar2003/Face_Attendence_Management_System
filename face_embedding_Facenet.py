import os
import pickle
import numpy as np
import cv2
from imutils import paths
from keras_facenet import FaceNet  # Import FaceNet model

# Paths to your dataset and output files
dataset = r"D:\PythonProject\Face_Recognition_DL\Face_recognition_Dataset"  # Path to your dataset
embeddingFile = "output/embeddings_facenet.pickle"

# Load FaceNet model
embedder = FaceNet()

def get_embedding(face_img):
    face_img = cv2.resize(face_img, (160, 160))  # Resize for FaceNet input
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)  # Convert to 4D tensor
    embedding = embedder.embeddings(face_img)  # Extract embedding
    return embedding[0] if embedding.shape == (1, 512) else None  # Ensure correct shape
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
    
    # Get FaceNet embedding
    embedding = get_embedding(image)
    
    # Store the embedding and person's name
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

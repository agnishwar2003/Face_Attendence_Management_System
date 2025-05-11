import os
import pickle
import numpy as np
import cv2
from imutils import paths

# Paths to your dataset and output files
dataset = r"D:\PythonProject\Face_Recognition_DL\Dataset_Face"  # Path to your dataset
embeddingFile = "Output_Models/embeddings_Openface08.pickle"
embeddingModel = r"D:\PythonProject\Face_Recognition_DL\model\openface.nn4.small2.v1.t7"  # Correct path to your model

# Verify model files and create directories if necessary
if not os.path.isdir("model"):
    os.makedirs("model")

if not os.path.isfile(embeddingModel):
    print(f"Embedding model file not found at: {embeddingModel}")
    raise FileNotFoundError("Embedding model file not found!")

# Load the facial embeddings model
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

# Ensure output directory exists
output_dir = os.path.dirname(embeddingFile)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
    name = imagePath.split(os.path.sep)[-2]  # The folder name is the person's name
    
    # Read and resize the image
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (96, 96))  # Resize image to 96x96 for embedding extraction

    # Prepare the face for embedding extraction
    faceBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    
    # Pass the face blob through the embedder to get the facial embedding
    embedder.setInput(faceBlob)
    vec = embedder.forward()

    # Store the embedding and the person's name (from the folder name)
    knownNames.append(name)
    knownEmbeddings.append(vec.flatten())
    total += 1

print(f"Total Embeddings: {total}")

# Save the embeddings to a file
data = {"embeddings": knownEmbeddings, "names": knownNames}

# Ensure the output directory exists and save the embeddings
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(embeddingFile, "wb") as f:
    f.write(pickle.dumps(data))

print("Process Completed")

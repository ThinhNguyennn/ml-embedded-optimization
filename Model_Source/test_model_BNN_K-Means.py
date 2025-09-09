import tensorflow as tf
import numpy as np
import cv2
import os
from collections import defaultdict
import pandas as pd
from PIL import Image
import time
from memory_profiler import profile
# Load the saved model
weights_params = np.load("BNN_model_cluster_29.npy", allow_pickle=True).item()
Wh, bh, Wo, bo = weights_params["Wh"], weights_params["bh"], weights_params["Wo"], weights_params["bo"]

def convert_to_matrix(bit_array, shape):
    flattened_array = bit_array.tolist()
    binary_matrix = np.array(bit_array).reshape(shape)
    return binary_matrix

Wh_shape = (512,784)
bh_shape = (1,512)
Wo_shape = (10,512)
bo_shape = (1,10)

Wh_matrix = convert_to_matrix(Wh, Wh_shape)
bh_matrix = convert_to_matrix(bh, bh_shape)
Wo_matrix = convert_to_matrix(Wo, Wo_shape)
bo_matrix = convert_to_matrix(bo, bo_shape)

def convert_weights_to_binary(weights):
  return np.where(weights != 0, 1, -1)

Wh = convert_weights_to_binary(Wh_matrix)
bh = convert_weights_to_binary(bh_matrix)
Wo = convert_weights_to_binary(Wo_matrix)
bo = convert_weights_to_binary(bo_matrix)
# =============================================================================
# Wo = weights_params["Wh"]
# bo = weights_params["bh"]
# Wh = weights_params["Wo"]
# bh = weights_params["bo"]
# =============================================================================

def preprocess_image(img):
    # Resize image to 28x28 pixels
    img = img.resize((28, 28))
    # Convert to grayscale
    img = img.convert('L')
    # Convert to numpy array
    img = np.array(img)
    # Flatten the image to match the input shape of the model
    img = np.reshape(img, (1, 784))
    # Convert pixel values to binary
    img[img <= 127] = 0
    img[img > 127] = 1
    return img

# Function to perform inference
#@profile
def predict_image(img, Wh, bh, Wo, bo):
    a = np.sign(np.dot(img, Wh.T) + bh)
    o = np.sign(np.dot(a, Wo.T) + bo)  
    return o

# Function to get files and labels
def get_file(folder_path, csv_path):
    dataset = defaultdict(list)
    data = pd.read_csv(csv_path)

    for index, row in data.iterrows():
        image_filename = row['filename']
        image_path = os.path.join(folder_path, image_filename)
        
        with Image.open(image_path) as img:
            img_gray = img.convert('L').resize((28, 28))
            labels = row[1:]
            img_labels = [label for label, value in labels.items() if value == 1]
            dataset['image'].append(preprocess_image(img_gray))
            dataset['label'].append(img_labels)
    
    return dataset

# Path to test folder and CSV file
test_folder_path = 'C:/Users/Admin/Desktop/Fianl Project Document/MNIST/test'
test_csv_path = 'C:/Users/Admin/Desktop/Fianl Project Document/MNIST/test/_classes.csv'

# Get images and labels from test folder
test_data = get_file(test_folder_path, test_csv_path)

# Initialize variables for accuracy calculation
total_samples = len(test_data['image'])
correct_predictions = 0

# Calculate accuracy
start_time = time.time()

for img, labels in zip(test_data['image'], test_data['label']):
    prediction = predict_image(img, Wh, bh, Wo, bo)
    predicted_digit = np.argmax(prediction)
    cleaned_labels = [label.strip() for label in labels]
    if str(predicted_digit) in cleaned_labels:
        correct_predictions += 1

accuracy = (correct_predictions / total_samples) * 100
end_time = time.time()
execution_time = end_time - start_time
print("Accuracy:", accuracy, "%")
print("Execution time:", execution_time, "seconds")


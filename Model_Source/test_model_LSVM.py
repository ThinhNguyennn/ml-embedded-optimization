import dill
import numpy as np
from PIL import Image
from collections import defaultdict
import pandas as pd
import os
from collections import Counter
from bitarray import bitarray
import time

test_folder_path = 'C:/Users/Admin/Desktop/Fianl Project Document/MNIST/test'
test_csv_path = 'C:/Users/Admin/Desktop/Fianl Project Document/MNIST/test/_classes.csv'

def get_file(folder_path, csv_path):
    dataset = defaultdict(list)
    data = pd.read_csv(csv_path)

    for index, row in data.iterrows():
        image_filename = row['filename']
        image_path = os.path.join(folder_path, image_filename)
        
        with Image.open(image_path) as img:
            img_gray = img.convert('L').resize((28, 28))
            #img_gray = img.convert('L')           
            labels = row[1:]
            img_labels = [label for label, value in labels.items() if value == 1]
            
            img_array = np.array(img_gray)
            for label in img_labels:
                dataset[label].append(img_array)

    return dataset

def normalize_images(dataset):
    normalized_dataset = defaultdict(list)
    for label, images in dataset.items():
        for image in images:
            normalized_image = image / 255.0
            normalized_dataset[label].append(normalized_image)
    return normalized_dataset

def load_model(filename):
    with open(filename, 'rb') as f:
        model = dill.load(f)
    return model

loaded_model_data = load_model('7741%.pkl')
representative_img = loaded_model_data['representative_img']
alpha = loaded_model_data['alpha']

def flatten(img):
    flatten_img = img.flatten().astype(bool)
    flatten_img = flatten_img.tolist()  
    return bitarray(flatten_img)

# Định nghĩa hàm popcount
def popcount(in1, in2):
    result_xor = in1 ^ in2
    return result_xor.count(True)

def classify(activity_data, alpha, representative_img, num_predictions):
    predictions = []
    
    for _ in range(num_predictions):
        min_error = float('inf')
        determined_label = None
        
        for label_data in representative_img:
            for cluster_data in representative_img[label_data]:
                binary_activity_data = np.where(activity_data > alpha[label_data][cluster_data], 1, 0)
                flatten_binary_activity_data = flatten(binary_activity_data)
                xor_result = popcount(flatten_binary_activity_data,representative_img[label_data][cluster_data])
                error_mean = np.sum(xor_result)
                if error_mean < min_error:
                    min_error = error_mean
                    determined_label = label_data
        predictions.append(determined_label)
        
    label_counts = Counter(predictions)
    most_common_label = label_counts.most_common(1)[0][0]

    if most_common_label == 0:
        return '0'
    elif most_common_label == 1:
        return '1'
    elif most_common_label == 2:
        return '2'
    elif most_common_label == 3:
        return '3'
    elif most_common_label == 4:
        return '4'
    elif most_common_label == 5:
        return '5'
    elif most_common_label == 6:
        return'6'
    elif most_common_label == 7:
        return'7'
    elif most_common_label == 8:
        return'8'
    elif most_common_label == 9:
        return'9'
    
test_dataset = get_file(test_folder_path, test_csv_path)
test_dataset_normalized = normalize_images(test_dataset)

def compute_accuracy(test_dataset_normalized, alpha, representative_img):
    correct_predictions = 0
    total_predictions = 0   
    for label, images in test_dataset_normalized.items():
        for image in images:
            predicted_label = classify(image, alpha, representative_img,1)
            if predicted_label == label:
                correct_predictions += 1
            total_predictions += 1
    
    accuracy = (correct_predictions / total_predictions)*100
    return accuracy
start_time = time.time()
accuracy = compute_accuracy(test_dataset_normalized,alpha, representative_img)
end_time = time.time()

execution_time = end_time - start_time
print("Accuracy:", accuracy)
print("Execution time:", execution_time, "seconds")





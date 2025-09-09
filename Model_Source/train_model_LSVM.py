# =============================================================================
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.cluster import KMeans
# import numpy as np
# from collections import defaultdict
# import matplotlib.pyplot as plt
# import dill
# from bitarray import bitarray
# import cv2
# from skimage.feature import hog
# import pandas as pd
# import os
# from PIL import Image
# 
# test_folder_path = 'C:/Users/Admin/Desktop/Fianl Project Document/MNIST/test'
# test_csv_path = 'C:/Users/Admin/Desktop/Fianl Project Document/MNIST/test/_classes.csv'
# 
# def get_file(folder_path, csv_path):
#     dataset = defaultdict(list)
#     data = pd.read_csv(csv_path)
# 
#     for index, row in data.iterrows():
#         image_filename = row['filename']
#         image_path = os.path.join(folder_path, image_filename)
#         
#         with Image.open(image_path) as img:
#             img_gray = img.convert('L').resize((28, 28))
#             #img_gray = img.convert('L')           
#             labels = row[1:]
#             img_labels = [label for label, value in labels.items() if value == 1]
#             
#             img_array = np.array(img_gray)
#             for label in img_labels:
#                 dataset[label].append(img_array)
# 
#     return dataset
# 
# def normalize_images(dataset):
#     normalized_dataset = defaultdict(list)
#     for label, images in dataset.items():
#         for image in images:
#             normalized_image = image / 255.0
#             normalized_dataset[label].append(normalized_image)
#     return normalized_dataset
# 
# def get_feature(img):
#     intensity = img.sum(axis=1)
#     intensity = intensity.sum(axis=0) / (img.shape[0] * img.shape[1])
#     return np.array([intensity])
# 
# def Preprocess(train_dataset, cluster_num):
#     preprocessed_dataset = defaultdict(lambda: defaultdict(list))
#     for label, images in train_dataset.items(): 
#         if len(images) >= cluster_num:
#             features = np.array([get_feature(img) for img in images])
#             model = KMeans(cluster_num, n_init=10)
#             model.fit(features)
#             preds = model.predict(features)       
#             for i, img in enumerate(images):
#                 cluster = preds[i]
#                 preprocessed_dataset[label][cluster].append(img)
#         else:
#             single_cluster = 0  
#             for img in images:
#                 preprocessed_dataset[label][single_cluster].append(img)   
#     return preprocessed_dataset
# 
# def flatten(img):
#     flatten_img = img.flatten().astype(bool)
#     flatten_img = flatten_img.tolist()  
#     return bitarray(flatten_img)
# 
# def popcount(in1, in2):
#     result_xor = in1 ^ in2
#     return result_xor.count(True)
# 
# def update(alpha, error, label, cluster):
#     error_mean = np.sum(error)
#     learning_rate = 0.00044
#     if error_mean > 34 and (alpha[label][cluster]).any() <= 0 :
#         alpha[label][cluster] += (learning_rate*error_mean)/3
#         a = (learning_rate*error_mean)/3
#         print("a++",a)
#     elif error_mean > 34 and (alpha[label][cluster]).any() > 0 :
#         alpha[label][cluster] -= (learning_rate*error_mean)/3
#         a = (learning_rate*error_mean)/3
#         print("a--",a)
#     else:
#         alpha[label][cluster] = alpha[label][cluster]
#     return alpha
# 
# def classify(activity_data, alpha, representative_img):
#     min_error = float('inf')
#     determined_label = None
#     for label_data in representative_img:
#         for cluster_data in representative_img[label_data]:
#             binary_activity_data = np.where(activity_data > alpha[label_data][cluster_data], 1, 0)
#             flatten_binary_activity_data = flatten(binary_activity_data)
#             xor_result = popcount(flatten_binary_activity_data,representative_img[label_data][cluster_data])
#             error_mean = np.sum(xor_result)
#             if error_mean < min_error:
#                 min_error = error_mean
#                 determined_label = label_data
#     if determined_label is not None:
#         return str(determined_label)
#     else:
#         return None
# 
# def compute_accuracy(test_dataset_normalized, alpha, representative_img):
#     correct_predictions = 0
#     total_predictions = 0   
#     for label, images in test_dataset_normalized.items():
#         label = label.strip()
#         for image in images:
#             predicted_label = classify(image, alpha, representative_img)
#             if predicted_label == label:
#                 correct_predictions += 1
#             total_predictions += 1
#     
#     accuracy = (correct_predictions / total_predictions)*100
#     return accuracy
# 
# def Train(binary_preprocessed_test_dataset, binary_preprocessed_train_dataset, num_epochs, test_dataset_normalized):            
#     alpha = defaultdict(lambda: defaultdict(list))
#     RP = defaultdict(lambda: defaultdict(list))        
#     R = defaultdict(lambda: defaultdict(list))
#     updated = defaultdict(lambda: defaultdict(list))
#     img_sum_dict = defaultdict(lambda: defaultdict(list))
#     avg_errors = []
#     accuracies = []
#     best_accuracy = 0
#     best_model_data = None  
#     for epoch in range(num_epochs): 
#         for label_data in binary_preprocessed_train_dataset:
#             for cluster_data in binary_preprocessed_train_dataset[label_data]:
#                 if not updated[label_data].get(cluster_data):
#                     img_sum = np.zeros((28, 28))
#                     count = 0
#                     for img in binary_preprocessed_train_dataset[label_data][cluster_data]:
#                         count += 1
#                         img_sum += img
#                     img_average = img_sum / count 
#                     alpha[label_data][cluster_data] = img_average.astype(np.float16)
#                     print("mean alpha", np.mean(alpha[label_data][cluster_data]))
#                     updated[label_data][cluster_data] = True
#                     img_sum_dict[label_data][cluster_data] = img_sum
#                 R[label_data][cluster_data] = img_sum_dict[label_data][cluster_data]
#                 R[label_data][cluster_data] = np.where(R[label_data][cluster_data] > alpha[label_data][cluster_data], 1, 0)
#                 RP[label_data][cluster_data] = flatten(R[label_data][cluster_data])
#                 for td in binary_preprocessed_test_dataset[label_data][cluster_data]:
#                     error = 0
#                     td = np.where(td > alpha[label_data][cluster_data], 1, 0)
#                     td_binary = flatten(td)
#                     error += popcount(RP[label_data][cluster_data], td_binary) 
#                     error_mean = np.sum(error)
#                 if updated[label_data].get(cluster_data):                           
#                     alpha = update(alpha, error, label_data, cluster_data)
#         avg_error = error_mean
#         avg_errors.append(avg_error)
#         accuracy = compute_accuracy(test_dataset_normalized, alpha, RP)        
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_model_data = {'representative_img': RP, 'alpha': alpha}
#             save_model(best_model_data, 'best_model.pkl')
#             print("Best model saved.")
#         print(f"Epoch {epoch+1} completed. Accuracy: {accuracy:.2f}%")
#         accuracies.append(accuracy)
#     print(f"Best accuracy achieved: {best_accuracy:.2f}%")
#     
#     return best_model_data, avg_errors, accuracies
# 
# 
# def save_model(model, filename):
#     with open(filename, 'wb') as file:
#         dill.dump(model, file)
#         
# def main():
#     mnist = tf.keras.datasets.mnist
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
# 
#     train_dataset = {i: [] for i in range(10)}
#     valid_dataset = {i: [] for i in range(10)}
#     test_dataset = get_file(test_folder_path, test_csv_path)
#     x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
#     
#     for img_array, label in zip(x_train, y_train):
#         train_dataset[label].append(img_array)
# 
#     for img_array, label in zip(x_valid, y_valid):
#         valid_dataset[label].append(img_array)
#     
#     train_dataset_normalized = normalize_images(train_dataset)
#     valid_dataset_normalized = normalize_images(valid_dataset)
#     test_dataset_normalized = normalize_images(test_dataset)
#     
#     cluster_num = 4
#     
#     preprocessed_train_dataset = Preprocess(train_dataset_normalized, cluster_num)
#     preprocessed_valid_dataset = Preprocess(valid_dataset_normalized, cluster_num)
#     
#     num_epochs = 50
#     best_model_data, avg_errors, accuracies = Train(preprocessed_valid_dataset, preprocessed_train_dataset, num_epochs, test_dataset_normalized)
#     
#     plt.plot(range(1, num_epochs + 1), avg_errors, label='Error')
#     plt.xlabel('Epoch')
#     plt.ylabel('Error')
#     plt.title('Error over Epochs')
#     plt.legend()
#     plt.show()
#     
#     plt.plot(range(1, num_epochs + 1), accuracies, label='Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy (%)')
#     plt.title('Accuracy over Epochs')
#     plt.legend()
#     plt.show()
# 
# if __name__ == "__main__":
#     main()
# 
# =============================================================================
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import dill
from bitarray import bitarray
import cv2
from skimage.feature import hog
import pandas as pd
import os
from PIL import Image

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

def get_feature(img):
    intensity = img.sum(axis=1)
    intensity = intensity.sum(axis=0) / (img.shape[0] * img.shape[1])
    return np.array([intensity])

def Preprocess(train_dataset, cluster_num):
    preprocessed_dataset = defaultdict(lambda: defaultdict(list))
    for label, images in train_dataset.items(): 
        if len(images) >= cluster_num:
            features = np.array([get_feature(img) for img in images])
            model = KMeans(cluster_num, n_init=10)
            model.fit(features)
            preds = model.predict(features)       
            for i, img in enumerate(images):
                cluster = preds[i]
                preprocessed_dataset[label][cluster].append(img)
        else:
            single_cluster = 0  
            for img in images:
                preprocessed_dataset[label][single_cluster].append(img)   
    return preprocessed_dataset

def flatten(img):
    flatten_img = img.flatten().astype(bool)
    flatten_img = flatten_img.tolist()  
    return bitarray(flatten_img)

def popcount(in1, in2):
    result_xor = in1 ^ in2
    return result_xor.count(True)

def update(alpha, error, label, cluster):
    error_mean = np.sum(error)
    learning_rate = 0.00044
    if error_mean > 34 and (alpha[label][cluster]).any() <= 0 :
        alpha[label][cluster] += (learning_rate*error_mean)/3
    elif error_mean > 34 and (alpha[label][cluster]).any() > 0 :
        alpha[label][cluster] -= (learning_rate*error_mean)/3
    else:
        alpha[label][cluster] = alpha[label][cluster]
    return alpha

def classify(activity_data, alpha, representative_img):
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
    if determined_label is not None:
        return str(determined_label)
    else:
        return None

def compute_accuracy(test_dataset_normalized, alpha, representative_img):
    correct_predictions = 0
    total_predictions = 0   
    for label, images in test_dataset_normalized.items():
        label = label.strip()
        for image in images:
            predicted_label = classify(image, alpha, representative_img)
            if predicted_label == label:
                correct_predictions += 1
            total_predictions += 1
    
    accuracy = (correct_predictions / total_predictions)*100
    return accuracy

def Train(binary_preprocessed_test_dataset, binary_preprocessed_train_dataset, num_epochs, test_dataset_normalized):            
    alpha = defaultdict(lambda: defaultdict(list))
    RP = defaultdict(lambda: defaultdict(list))        
    R = defaultdict(lambda: defaultdict(list))
    updated = defaultdict(lambda: defaultdict(list))
    img_sum_dict = defaultdict(lambda: defaultdict(list))
    avg_errors = []
    accuracies = []
    best_accuracy = 0
    best_model_data = None  
    
    alpha_means = defaultdict(lambda: defaultdict(list))  # To track mean alpha values
    
    for epoch in range(num_epochs): 
        for label_data in binary_preprocessed_train_dataset:
            for cluster_data in binary_preprocessed_train_dataset[label_data]:
                if not updated[label_data].get(cluster_data):
                    img_sum = np.zeros((28, 28))
                    count = 0
                    for img in binary_preprocessed_train_dataset[label_data][cluster_data]:
                        count += 1
                        img_sum += img
                    img_average = img_sum / count 
                    alpha[label_data][cluster_data] = img_average.astype(np.float16)
                    #print("mean alpha", np.mean(alpha[label_data][cluster_data]))
                    updated[label_data][cluster_data] = True
                    img_sum_dict[label_data][cluster_data] = img_sum
                R[label_data][cluster_data] = img_sum_dict[label_data][cluster_data]
                R[label_data][cluster_data] = np.where(R[label_data][cluster_data] > alpha[label_data][cluster_data], 1, 0)
                RP[label_data][cluster_data] = flatten(R[label_data][cluster_data])
                for td in binary_preprocessed_test_dataset[label_data][cluster_data]:
                    error = 0
                    td = np.where(td > alpha[label_data][cluster_data], 1, 0)
                    td_binary = flatten(td)
                    error += popcount(RP[label_data][cluster_data], td_binary) 
                    error_mean = np.sum(error)
                if updated[label_data].get(cluster_data):                           
                    alpha = update(alpha, error, label_data, cluster_data)
        
        # Track mean alpha values for each cluster
        for label in alpha:
            for cluster in alpha[label]:
                alpha_means[label][cluster].append(np.max(alpha[label][cluster]))
        
        avg_error = error_mean
        avg_errors.append(avg_error)
        accuracy = compute_accuracy(test_dataset_normalized, alpha, RP)        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_data = {'representative_img': RP, 'alpha': alpha}
            save_model(best_model_data, 'best_model.pkl')
            print("Best model saved.")
        print(f"Epoch {epoch+1} completed. Accuracy: {accuracy:.2f}%")
        accuracies.append(accuracy)
    print(f"Best accuracy achieved: {best_accuracy:.2f}%")
    
    return best_model_data, avg_errors, accuracies, alpha_means


def save_model(model, filename):
    with open(filename, 'wb') as file:
        dill.dump(model, file)
        
def main():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    train_dataset = {i: [] for i in range(10)}
    valid_dataset = {i: [] for i in range(10)}
    test_dataset = get_file(test_folder_path, test_csv_path)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    
    for img_array, label in zip(x_train, y_train):
        train_dataset[label].append(img_array)

    for img_array, label in zip(x_valid, y_valid):
        valid_dataset[label].append(img_array)
    
    train_dataset_normalized = normalize_images(train_dataset)
    valid_dataset_normalized = normalize_images(valid_dataset)
    test_dataset_normalized = normalize_images(test_dataset)
    
    cluster_num = 4
    
    preprocessed_train_dataset = Preprocess(train_dataset_normalized, cluster_num)
    preprocessed_valid_dataset = Preprocess(valid_dataset_normalized, cluster_num)
    
    num_epochs = 50
    best_model_data, avg_errors, accuracies, alpha_means = Train(preprocessed_valid_dataset, preprocessed_train_dataset, num_epochs, test_dataset_normalized)
    
    plt.plot(range(1, num_epochs + 1), avg_errors, label='Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Error over Epochs')
    plt.legend()
    plt.show()
    
    plt.plot(range(1, num_epochs + 1), accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.show()
    
    fig, axes = plt.subplots(5, 2, figsize=(12, 18))

    # Lặp qua mỗi label và vẽ đồ thị vào mỗi ô của lưới đồ thị
    for i, label in enumerate(alpha_means):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        for cluster in alpha_means[label]:
            ax.plot(range(1, num_epochs + 1), alpha_means[label][cluster], label=f'Cluster {cluster}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Alpha Value')
        ax.set_title(f'Mean Alpha Value over Epochs for Label {label}')
        ax.legend()
    
    # Đảm bảo rằng không có ô nào không được sử dụng
    for i in range(len(alpha_means), 5 * 2):
        row = i // 2
        col = i % 2
        axes[row, col].axis('off')
    
    plt.tight_layout()  # Đảm bảo không có chồng chéo giữa các ô đồ thị
    plt.show()

if __name__ == "__main__":
    main()


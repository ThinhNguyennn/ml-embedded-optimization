import dill
import numpy as np
from bitarray import bitarray
import cv2

def load_model(filename):
    with open(filename, 'rb') as f:
        model = dill.load(f)
    return model

loaded_model_data = load_model('Model_LSVM.pkl')
representative_img = loaded_model_data['representative_img']
alpha = loaded_model_data['alpha']

def preprocess_image(image):
    return np.array(image) / 255.0

def flatten(img):
    flatten_img = img.flatten().astype(bool)
    flatten_img = flatten_img.tolist()  
    return bitarray(flatten_img)

def popcount(in1, in2):
    result_xor = in1 ^ in2
    return result_xor.count(True)

def predict_label_with_LSVM(frame, alpha=alpha, representative_img=representative_img):
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    image_gray = preprocess_image(image)
    min_error = float('inf')
    determined_label = None
    
    for label_data in representative_img:
        for cluster_data in representative_img[label_data]:
            binary_activity_data = np.where(image_gray > alpha[label_data][cluster_data], 1, 0)
            flatten_binary_activity_data = flatten(binary_activity_data)
            xor_result = popcount(flatten_binary_activity_data,representative_img[label_data][cluster_data])
            error_mean = np.sum(xor_result)
            if error_mean < min_error:
                min_error = error_mean
                determined_label = label_data
                print("label",determined_label)
    if determined_label == 0:
        return '0'
    elif determined_label == 1:
        return '1'
    elif determined_label == 2:
        return '2'
    elif determined_label == 3:
        return '3'
    elif determined_label == 4:
        return '4'
    elif determined_label == 5:
        return '5'
    elif determined_label == 6:
        return'6'
    elif determined_label == 7:
        return'7'
    elif determined_label == 8:
        return'8'
    elif determined_label == 9:
        return'9'
    





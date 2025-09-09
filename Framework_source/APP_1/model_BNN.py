import tensorflow as tf
import numpy as np
import cv2
# Load the saved model
weights_params = np.load("Model_BNN.npy", allow_pickle=True).item()
Wh, bh, Wo, bo = weights_params["Wh"], weights_params["bh"], weights_params["Wo"], weights_params["bo"]

def convert_to_matrix(bit_array, shape):
    flattened_array = bit_array.tolist()
    binary_matrix = np.array(flattened_array).reshape(shape)
    return binary_matrix

Wh_shape = (512, 784)
bh_shape = (1, 512)
Wo_shape = (10, 512)
bo_shape = (1, 10)

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

def preprocess_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(img_gray)
    img = np.reshape(img, (1, 784))
    img[img <= 127] = 0
    img[img > 127] = 1
    return img

def predict_image(img):
    a = np.sign(np.dot(img, Wh.T) + bh)
    o = np.sign(np.dot(a, Wo.T) + bo)
    return o

def predict_label_with_BNN(frame):
    image = preprocess_image(frame)
    prediction = predict_image(image)
    predicted_digit = np.argmax(prediction)
    print("label",predicted_digit)
    return predicted_digit
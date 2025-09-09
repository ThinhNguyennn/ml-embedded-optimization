import tensorflow as tf
import numpy as np
import cv2

# Load a pre-trained MNIST model without compiling it
model = tf.keras.models.load_model("Model_CNN.h5", compile=False)
# Compile the model with a standard optimizer
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Function to preprocess image
def preprocess_image(image):
    return np.array(image) / 255.0

# Function to predict and display label
def predict_label_with_CNN(frame, threshold=0.5):
    # Preprocess image
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    image_gray = preprocess_image(image)

    # Reshape image to fit model input
    image_array = image_gray.reshape(1, 28, 28, 1)

    # Predict the label
    prediction = model.predict(image_array)
    confidence = np.max(prediction)
    label = np.argmax(prediction)
    print("label",label)
    # Check if prediction confidence is above threshold
    if confidence >= threshold:
        return label
    else:
        return "invalid"

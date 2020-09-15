import tensorflow.keras as keras
import tensorflow.keras.utils as utils
import cv2
from preprocess import find_images_min_sizes, transform_image_to_sizes, IRIS_PROCESSED_DATASET_PATH
import numpy as np
from PIL import Image

model = keras.models.load_model('model.h5')

X_pred = np.array([np.asarray(cv2.imread('datasets/MMU Iris Database/1/left/aeval1.bmp', cv2.IMREAD_GRAYSCALE))])
X_pred = X_pred / 255
X_pred = X_pred.reshape(-1, X_pred.shape[1], X_pred.shape[2], 1)

y_pred = model.predict_classes(X_pred)

print(y_pred)
import tensorflow.keras as keras
import os, shutil
import re
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

original_dataset_dir = './iris/dataset'
base_dir = './iris/demo/'
os.mkdir(base_dir)

iris_segmented = base_dir + 'iris_dataset'
os.mkdir(iris_segmented)

filename = ''

irisList = os.listdir(original_dataset_dir)

for iris in irisList:
    filename = original_dataset_dir + iris

    grad_iris_img = cv2.imread(filename, 0)
    iris_img = cv2.imread(filename)

    crop_iris = grad_iris_img[:, :]
    crop_orig_iris = iris_img[:, :]
    crop_iris = cv2.medianBlur(crop_iris, 5)

    if iris == 'masl2.tmp':
        circles = cv2.HoughCircles(crop_iris, cv2.HOUGH_GRADIENT, 1.1, 155,
                                   param1=50, param2=30, minRadius=35, maxRadius=70)
    else:
        circles = cv2.HoughCircles(crop_iris, cv2.HOUGH_GRADIENT, 1.0, 155,
                                   param1=50, param2=30, minRadius=35, maxRadius=70)

    if circles is None:
        circles = np.uint16(np.around(circles))
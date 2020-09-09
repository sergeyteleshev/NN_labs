from PIL import Image
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, utils
import os, shutil
import re
import glob
import cv2
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

from preprocess import IRIS_EYE_DATASET_PATH, NAMES_FEMALE_DATASET_PATH, NAMES_MALE_DATASET_PATH, MY_EYE_PATH, IRIS_PROCESSED_DATASET_PATH, OUTPUT_CSV_PATH


def train():
    df = pd.read_csv(OUTPUT_CSV_PATH)
    X_train = np.array([np.asarray(Image.open(im)) for im in df['iris_eye_picture_path']])
    X_train = X_train / 255
    X_train = list(X_train)

    le = preprocessing.LabelEncoder()
    labels = list(set(df['names']))
    le.fit(labels)

    y_train = le.transform(df['names'])
    y_train_onehot = utils.to_categorical(y_train)
    print(y_train_onehot)

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(None, None, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train_onehot, epochs=10)

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')


if __name__ == '__main__':
    train()
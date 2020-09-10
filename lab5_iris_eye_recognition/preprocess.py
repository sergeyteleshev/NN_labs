import os, sys, math
import numpy as np
import pandas as pd
import cv2

IRIS_EYE_DATASET_PATH = 'datasets/Dataset/'
NAMES_FEMALE_DATASET_PATH = 'datasets/names/female.txt'
NAMES_MALE_DATASET_PATH = 'datasets/names/male.txt'
MY_EYE_PATH = 'datasets/sergeyteleshev_eye/sergeyteleshev_croped.jpg'
IRIS_PROCESSED_DATASET_PATH = 'datasets/iris processed/'
OUTPUT_CSV_PATH = "output/iris_eyes_dataset.csv"


def find_images_min_sizes(images_path):
    min_width = sys.maxsize
    min_height = sys.maxsize
    min_channels = sys.maxsize

    for dirname, _, filenames in os.walk(images_path):
        for filename in filenames:
            if filename[-3:] == 'png' or filename[-3:] == 'jpg' or filename[-4:] == 'jpeg':
                img = cv2.imread(images_path + filename, 1)
                height, width, channels = img.shape
                if height <= min_height and width <= min_width and channels <= min_channels:
                    min_width = width
                    min_height = height
                    min_channels = channels

    return min_width, min_height, min_channels


def transform_image_to_sizes(image_path, min_width, min_height, min_channels):
    img = cv2.imread(image_path, 1)
    height, width, channels = img.shape

    top_edge = math.ceil((height - min_height) / 2)
    bottom_edge = height - math.floor((height - min_height) / 2)
    left_edge = math.ceil((width - min_width) / 2)
    right_edge = width - math.ceil((width - min_width) / 2)

    cropped_img = img[top_edge: bottom_edge, left_edge: right_edge]

    return cropped_img


def preprocess():
    names = []
    distances = []
    eye_position = []
    eyes_pictures_path = []

    min_width, min_height, min_channels = find_images_min_sizes(IRIS_EYE_DATASET_PATH)

    for dirname, _, filenames in os.walk(IRIS_EYE_DATASET_PATH):
        for filename in filenames:
            if filename[-3:] == 'png' or filename[-3:] == 'jpg' or filename[-4:] == 'jpeg':
                cropped_image = transform_image_to_sizes(IRIS_EYE_DATASET_PATH + filename, min_width, min_height,
                                                         min_channels)
                save_path = IRIS_PROCESSED_DATASET_PATH + filename
                cv2.imwrite(save_path, cropped_image)

                names.append(filename.split(' ')[0])
                eyes_pictures_path.append(save_path)
                distances.append(filename.split(' ')[1])
                eye_position.append(filename.split(' ')[2][:-4])

    df = pd.DataFrame(data={'names': names, 'iris_eye_picture_path': eyes_pictures_path, 'eye_position': eye_position})

    df.dropna(axis=0, inplace=True)

    print(df.tail())
    df.to_csv(OUTPUT_CSV_PATH)


if __name__ == '__main__':
    preprocess()
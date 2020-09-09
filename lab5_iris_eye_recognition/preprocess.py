import os
import numpy as np
import pandas as pd
import cv2
from detect_iris import detect_iris, detect_pupil

IRIS_EYE_DATASET_PATH = 'datasets/Dataset/'
NAMES_FEMALE_DATASET_PATH = 'datasets/names/female.txt'
NAMES_MALE_DATASET_PATH = 'datasets/names/male.txt'
MY_EYE_PATH = 'datasets/sergeyteleshev_eye/sergeyteleshev_croped.jpg'
IRIS_PROCESSED_DATASET_PATH = 'datasets/iris processed/'
OUTPUT_CSV_PATH = "output/iris_eyes_dataset.csv"

# with open(NAMES_MALE_DATASET_PATH) as f:
#     male_names = [name.split('\n')[0] for name in list(f)]
#
# with open(NAMES_FEMALE_DATASET_PATH) as f:
#     female_names = [name.split('\n')[0] for name in list(f)]

# names = np.unique(np.concatenate((male_names, female_names)))
names = []
distances = []
eye_position = []
eyes_pictures_path = []
# iris_eye_cropped_picture_path = []

# print('Now we have {} unique names'.format(len(names)))


for dirname, _, filenames in os.walk(IRIS_EYE_DATASET_PATH):
    for filename in filenames:
        if filename[-3:] == 'png' or filename[-3:] == 'jpg' or filename[-4:] == 'jpeg':
            names.append(filename.split(' ')[0])
            eyes_pictures_path.append(IRIS_EYE_DATASET_PATH + filename)
            distances.append(filename.split(' ')[1])
            eye_position.append(filename.split(' ')[2][:-4])

            # img_name = filename
            # img = cv2.imread(IRIS_EYE_DATASET_PATH + filename, 1)
            # pupil_center, pupil_radius = detect_pupil(img)
            #
            # if pupil_center is None or pupil_radius is None:
            #     iris_eye_cropped_picture_path.append(np.nan)
            #     continue
            #
            # iris_img = detect_iris(img, (pupil_center[0], pupil_center[1], pupil_radius))
            #
            # if iris_img is None:
            #     iris_eye_cropped_picture_path.append(np.nan)
            #     continue
            #
            # iris_img_path = IRIS_PROCESSED_DATASET_PATH + img_name
            # cv2.imwrite(iris_img_path, iris_img)
            # iris_eye_cropped_picture_path.append(iris_img_path)

# print('And we have {} unique pictures'.format(len(eyes_pictures_path)))
#
# min_samples = min([len(eyes_pictures_path), len(names)])

# df = pd.DataFrame(data={'names': names[:min_samples], 'iris_eye_picture_path': eyes_pictures_path[:min_samples]})
# df.loc[len(df)] = ['Sergeyteleshev'] + [MY_EYE_PATH]
# df['iris_eye_cropped_pucture_path'] = iris_eye_cropped_picture_path

df = pd.DataFrame(data={'names': names, 'iris_eye_picture_path': eyes_pictures_path, 'eye_position': eye_position})

df.dropna(axis=0, inplace=True)

print(df.tail())
df.to_csv(OUTPUT_CSV_PATH)
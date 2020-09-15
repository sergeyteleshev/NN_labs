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

OUTPUT_MMU_CSV_PATH = 'output/MMU_dataset.csv'
OUTPUT_MMU_CUT_CSV_PATH = 'output/MMU_cut_dataset.csv'
OUTPUT_TEST_MMU_CUT_CSV_PATH = 'output/test_MMU_cut_dataset.csv'

MMU_DATASET_PATH = 'datasets/MMU Iris Database'
MMU_CUT_DATASET_PATH = 'datasets/MMU'
MMU_TEST_CUT_DATASET_PATH = 'datasets/MMU2'


def find_images_min_sizes(images_path):
    min_width = sys.maxsize
    min_height = sys.maxsize
    min_channels = sys.maxsize

    for dirname, _, filenames in os.walk(images_path):
        for filename in filenames:
            if filename[-3:] == 'png' or filename[-3:] == 'jpg' or filename[-4:] == 'jpeg':
                img = cv2.imread(images_path + filename, 1)
                height, width, channels = img.shape

                if height <= min_height:
                    min_height = height

                if width <= min_width:
                    min_width = width

                if channels <= min_channels:
                    min_channels = channels

    return min_width, min_height, min_channels


def transform_image_to_sizes(image_path, min_width, min_height, min_channels):
    img = cv2.imread(image_path, 1)
    height, width, channels = img.shape

    top_edge = math.floor((height - min_height) / 2)
    bottom_edge = height - math.floor((height - min_height) / 2)
    left_edge = math.floor((width - min_width) / 2)
    right_edge = width - math.floor((width - min_width) / 2)

    cropped_img = img[top_edge: bottom_edge, left_edge: right_edge]

    return cropped_img


def transform_image_to_current_size(image_path, min_width, min_height, min_channels):
    img = cv2.imread(image_path, 1)
    cropped_img = img[0: min_height, 0: min_width]

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
                cropped_image = cv2.resize(cropped_image, (min_width, min_height))

                cv2.imwrite(save_path, cropped_image)

                names.append(filename.split(' ')[0])
                eyes_pictures_path.append(save_path)
                distances.append(filename.split(' ')[1])
                eye_position.append(filename.split(' ')[2][:-4])

    df = pd.DataFrame(data={'names': names, 'iris_eye_picture_path': eyes_pictures_path, 'eye_position': eye_position})

    df.dropna(axis=0, inplace=True)

    print(df.tail())
    df.to_csv(OUTPUT_CSV_PATH)


def preprocess_mmu_dataset():
    labels = []
    for x in os.walk(MMU_DATASET_PATH):
        labels = x[1]
        break

    files_path = []
    persons = []
    persons_eye_position = []

    for label_dir in labels:
        labels_path = os.path.join(MMU_DATASET_PATH, label_dir)
        for address, dirs, files in os.walk(labels_path):
            for dir in dirs:
                dir_path = os.path.relpath(os.path.join(labels_path, dir))
                eye_position = dir_path.split('\\')[-1]
                person = dir_path.split('\\')[-2]
                for ad, dr, fl in os.walk(dir_path):
                    for file in fl:
                        file_path = os.path.join(person, eye_position, file)
                        if file_path[-3:] == "bmp":
                            persons.append(person)
                            persons_eye_position.append(eye_position)
                            files_path.append(file_path.replace("\\", "/"))

    df = pd.DataFrame(data={'person': persons, 'eye_position': persons_eye_position, 'file_path': files_path})
    df.dropna(axis=0, inplace=True)

    df.to_csv(OUTPUT_MMU_CSV_PATH)

    return df


def preprocess_mmu_dataset_with_tests():
    labels = []
    for x in os.walk(MMU_CUT_DATASET_PATH):
        labels = x[1]
        break

    files_path = []
    persons = []

    test_persons = []
    test_files_path = []

    for label_dir in labels:
        person_eyes_path = os.path.join(MMU_CUT_DATASET_PATH, label_dir).replace("\\", "/")
        for ad, dr, fl in os.walk(person_eyes_path):
            for file in fl:
                person = person_eyes_path.split("/")[-1]
                file_path = os.path.join(person, file)
                if file_path[-3:] == "bmp":
                    persons.append(person)
                    files_path.append(file_path.replace("\\", "/"))

        test_person_eyes_path = os.path.join(MMU_TEST_CUT_DATASET_PATH, label_dir).replace("\\", "/")
        for ad, dr, fl in os.walk(test_person_eyes_path):
            for file in fl:
                test_person = test_person_eyes_path.split("/")[-1]
                test_file_path = os.path.join(test_person, file)
                if test_file_path[-3:] == "bmp":
                    test_persons.append(test_person)
                    test_files_path.append(test_file_path.replace("\\", "/"))

    df = pd.DataFrame(data={'person': persons,
                            'file_path': files_path})
    df.dropna(axis=0, inplace=True)

    test_df = pd.DataFrame(data={'person': test_persons, 'file_path': test_files_path})
    test_df.dropna(axis=0, inplace=True)

    df.to_csv(OUTPUT_MMU_CUT_CSV_PATH)
    test_df.to_csv(OUTPUT_TEST_MMU_CUT_CSV_PATH)

    return df, test_df


if __name__ == '__main__':
    df = preprocess_mmu_dataset_with_tests()

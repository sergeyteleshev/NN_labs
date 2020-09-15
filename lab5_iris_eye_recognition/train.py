from tensorflow.keras import layers, models, utils
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from preprocess import preprocess_mmu_dataset_with_tests, MMU_CUT_DATASET_PATH, MMU_TEST_CUT_DATASET_PATH


def get_train_and_test_data():
    df, df_test = preprocess_mmu_dataset_with_tests()

    X_train = np.array([cv2.imread(os.path.join(MMU_CUT_DATASET_PATH, img_path), cv2.IMREAD_GRAYSCALE) for img_path in df['file_path']])
    X_test = np.array([cv2.imread(os.path.join(MMU_TEST_CUT_DATASET_PATH, img_path), cv2.IMREAD_GRAYSCALE) for img_path in df_test['file_path']])

    y_train = np.array(df['person']).astype('int32')
    y_test = np.array(df_test['person']).astype('int32')

    return X_train, X_test, y_train, y_test


def train():
    X_train, X_test, y_train, y_test = get_train_and_test_data()
    targets = set(y_train)

    input_shape = X_train[0].shape

    X_train = X_train / 255
    X_test = X_test / 255

    X_train = X_train.reshape(-1, input_shape[0], input_shape[1], 1)
    X_test = X_test.reshape(-1, input_shape[0], input_shape[1], 1)

    y_train = utils.to_categorical(y_train)
    y_test = utils.to_categorical(y_test)

    print(X_train.shape)
    print(X_test.shape)

    model = models.Sequential()  # add model layers
    model.add(layers.Conv2D(32, kernel_size=(5, 5),
                            activation='relu',
                            input_shape=X_train.shape[1:]))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # add second convolutional layer with 20 filters
    model.add(layers.Conv2D(64, (5, 5), activation='relu'))

    # add 2D pooling layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # flatten data
    model.add(layers.Flatten())

    # add a dense all-to-all relu layer
    model.add(layers.Dense(1024, activation='relu'))

    # apply dropout with rate 0.5
    model.add(layers.Dropout(0.5))

    # soft-max layer
    model.add(layers.Dense(len(targets), activation='softmax'))
    # compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=1)

    model.save('model.h5')

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    model.evaluate(X_test, y_test)


if __name__ == '__main__':
    train()

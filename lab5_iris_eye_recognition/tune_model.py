from tensorflow.keras import layers, models, utils
import numpy as np
import cv2
import os
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import pickle
import time

from preprocess import preprocess_mmu_dataset_with_tests, MMU_CUT_DATASET_PATH, MMU_TEST_CUT_DATASET_PATH

LOG_DIR = f"{int(time.time())}"


def get_train_and_test_data():
    df, df_test = preprocess_mmu_dataset_with_tests()

    x_train = np.array([cv2.imread(os.path.join(MMU_CUT_DATASET_PATH, img_path), cv2.IMREAD_GRAYSCALE) for img_path in
                        df['file_path']])
    x_test = np.array(
        [cv2.imread(os.path.join(MMU_TEST_CUT_DATASET_PATH, img_path), cv2.IMREAD_GRAYSCALE) for img_path in
         df_test['file_path']])

    y_train = np.array(df['person']).astype('int32')
    y_test = np.array(df_test['person']).astype('int32')

    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = get_train_and_test_data()
targets = set(y_train)
num_output = len(targets)

x_train = x_train / 255
x_test = x_test / 255

input_shape = x_train[0].shape

x_train = x_train.reshape(-1, input_shape[0], input_shape[1], 1)
x_test = x_test.reshape(-1, input_shape[0], input_shape[1], 1)

y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

print(x_train.shape)
print(x_test.shape)


def build_model(hp):
    model = models.Sequential()  # add model layers
    model.add(layers.Conv2D(hp.Int("input_units", min_value=32, max_value=256, step=32), kernel_size=(5, 5),
                            activation='relu',
                            input_shape=x_train.shape[1:]))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # add second convolutional layer with 20 filters
    for i in range(hp.Int("n_layers", 1, 4)):
        model.add(
            layers.Conv2D(hp.Int(f"conv_{i}_units", min_value=32, max_value=256, step=32), (5, 5), activation='relu'))

    # add 2D pooling layer
    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # flatten data
    model.add(layers.Flatten())

    # add a dense all-to-all relu layer
    model.add(layers.Dense(1024, activation='relu'))

    # apply dropout with rate 0.5
    model.add(layers.Dropout(0.5))

    # soft-max layer
    model.add(layers.Dense(num_output, activation='softmax'))
    # compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

    # model = build_model(x_train.shape[1:], num_output)


tuner = RandomSearch(build_model,
                     objective="val_accuracy",
                     max_trials=1,
                     executions_per_trial=1,
                     directory=LOG_DIR)

tuner.search(x=x_train, y=y_train, epochs=1, validation_data=(x_test, y_test))

print(tuner.get_best_hyperparameters()[0].values)
# with open(f"tuner_{int(time.time())}.pkl", "wb") as f:
#     pickle.dump(tuner, f)
#
# tuner = pickle.load()

# history = model.fit(x_train, y_train, epochs=1)
#
# model.save('model.h5')
#
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['loss'], label='loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
#
# model.evaluate(x_test, y_test)

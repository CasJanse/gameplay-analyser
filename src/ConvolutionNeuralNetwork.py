from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Conv3D
from DataLoader import DataLoader
import cv2
import sys
import numpy as np
from inspect import currentframe, getframeinfo
import json


# region Data cleaning
def normalise_video_data(video):
    video_array = np.array(video)
    new_video_array = video_array.astype(float) * (1.0/256.0)
    new_video_array[new_video_array == 0.] = 0.001
    return new_video_array


def clean_data(data, data_type, chunk_size=5, overlap_size=1):
    # Return single frames
    if data_type == 1:
        chunk_size = 1
        chunks = list(divide_list_into_chunks(data, chunk_size))
        return chunks

    # Return a list with chunks of frames
    elif data_type == 2:
        if len(data) % chunk_size > 0:
            data = data[: -(len(data) % chunk_size)]
        chunks = list(divide_list_into_chunks(data, chunk_size))
        return chunks

    # Return a list with chunks of frames that overlap previous chunks
    elif data_type == 3:
        offset_size = chunk_size - overlap_size
        chunks = list(divide_list_into_chunks(data, chunk_size, True, offset_size))
        # Remove the last few incomplete chunks from the end of the list
        chunks = chunks[:-(max(int(chunk_size / offset_size), 2))]
        return chunks
    else:
        print("Data type not recognised")
    return None


def divide_list_into_chunks(list, chunk_size, overlap=False, overlap_size=1):
    step_size = chunk_size
    if overlap:
        step_size = overlap_size
    for i in range(0, len(list), step_size):
        yield list[i:i + chunk_size]


def transform_data_to_equal_length(video_data, input_data):
    if len(video_data) > len(input_data):
        video_data = video_data[: len(input_data)]
    if len(input_data) > len(video_data):
        input_data = input_data[: len(video_data)]
    return video_data, input_data


def categorise_input_data(input_data, input_format=1):
    categorised_input_data = np.empty([len(input_data), input_keys_amount * current_input_size_modifier])
    temp_categorised_input_data = np.empty([len(input_data), current_input_size_modifier, input_keys_amount])

    for i, chunk in enumerate(input_data):
        for j, data in enumerate(chunk):
            if input_format == 3:
                chunk_number = 0
            else:
                chunk_number = j

            if "up" in data:
                temp_categorised_input_data[i][chunk_number][0] = 1
            if "down" in data:
                temp_categorised_input_data[i][chunk_number][1] = 1
            if "left" in data:
                temp_categorised_input_data[i][chunk_number][2] = 1
            if "right" in data:
                temp_categorised_input_data[i][chunk_number][3] = 1
            if "z" in data:
                temp_categorised_input_data[i][chunk_number][4] = 1
        categorised_input_data[i] = temp_categorised_input_data[i].flatten()
    return categorised_input_data
# endregion


# region Run CNN models
def run_model_1(base_model, x_train, y_train, x_test, y_test):
    # region Model 1
    model_1 = build_model_1(base_model)

    # Compile model using accuracy to measure model performance
    model_1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    train_model(model_1, x_train, x_test, y_train, y_test)
    # endregion


def run_model_2(base_model, x_train, y_train, x_test, y_test):
    # region Model 2
    model_2 = build_model_2(base_model)

    # Compile model using accuracy to measure model performance
    model_2.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    train_model(model_2, x_train, x_test, y_train, y_test)
    # endregion


def run_model_3(base_model, x_train, y_train, x_test, y_test):
    # region Model 3
    model_3 = build_model_3(base_model)

    # Compile model using accuracy to measure model performance
    model_3.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    train_model(model_3, x_train, x_test, y_train, y_test)
    # endregion


def train_model(model, x_train, x_test, y_train, y_test):
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)

    save_model(model, history)
# endregion


# region Build CNN models
def build_model_1(model):
    model.add(Dense(64, activation="relu", input_shape=(current_chunk_size, 95, 103, 3)))
    model.add(Flatten())
    model.add(Dense(input_keys_amount * current_input_size_modifier, activation="softmax"))
    return model


def build_model_2(model):
    model.add(Conv3D(64, kernel_size=current_chunk_size, activation="relu", input_shape=(current_chunk_size, 95, 103, 3), data_format="channels_last"))
    model.add(Flatten())
    model.add(Dense(input_keys_amount * current_input_size_modifier, activation="softmax"))
    return model


def build_model_3(model):
    model.add(Conv3D(32, kernel_size=current_chunk_size, activation="relu", input_shape=(current_chunk_size, 95, 103, 3), data_format="channels_last"))
    model.add(Conv3D(18, kernel_size=1, activation="relu"))
    model.add(Flatten())
    model.add(Dense(input_keys_amount * current_input_size_modifier, activation="softmax"))
    return model
# endregion


# region Automation of model execution
def set_global_variables(model_code):

    global current_chunk_size
    global current_model
    global current_data_format
    global current_input_format
    global current_input_size_modifier
    global current_model_code

    current_model_code = model_code
    model_code = split(model_code)
    current_data_format = int(model_code[0])
    current_input_format = int(model_code[1])
    current_model = int(model_code[2])
    current_chunk_size = int(model_code[3])
    current_input_size_modifier = 1 if int(model_code[1]) == 3 else current_chunk_size


def get_chunk_size(value):
    if value == 1:
        return 1
    elif value == 2 or value == 3:
        return 3


def split(word):
    return [char for char in str(word)]


def save_model(model, history):
    model.save("../../networks/{}-20-08.h5".format(current_model_code))

    with open("../../networks/{}-20-08_history.json".format(current_model_code), "w+") as file:
        json.dump(str(history.history), file)

# endregion


# region DEBUG
def show_video(video_data, input_data, result=None, playback_speed=10):
    for i, chunk in enumerate(video_data):
        for j, frame in enumerate(chunk):
            print("Recorded inputs: {}".format(input_data[i]))
            if result is not None:
                print("Model prediction: {}".format(result[i]))
            cv2.imshow("Test", frame)

            if cv2.waitKey(playback_speed) & 0xFF == ord("q"):
                pass
    cv2.destroyAllWindows()
    terminate_program(getframeinfo(currentframe()).lineno, "Debug video ended")


def terminate_program(line_number=0, message="Program terminated (DEBUG)"):
    sys.exit(message + "\nLine: {}".format(line_number))
# endregion DEBUG


def train_networks(models_to_run):
    for model_code in models_to_run:
        video_data, input_data = get_data_from_video("output_8.avi", "inputs_8.csv", 20, model_code)

        print(video_data.shape)
        print(input_data.shape)

        # show_video(video_data, input_data)
        # terminate_program(getframeinfo(currentframe()).lineno)

        x_train, x_test, y_train, y_test = convert_data_to_train_test_batches(video_data, input_data, 0.8)

        print("Data format: {} - Input format: {} - Model: {} - Chunk size: {}".format(current_data_format, current_input_format, current_model, current_chunk_size))

        # region Mnist test
        # # Download mnist data and slit into train and test sets
        # (x_train, y_train), (x_test, y_test) = mnist.load_data()
        #
        # print(x_train.shape)
        #
        # # Reshape data to fit the model
        # x_train = x_train.reshape(60000, 28, 28, 1)
        # x_test = x_test.reshape(10000, 28, 28, 1)
        #
        # print(x_train.shape)
        #
        # # One-shot encode target column
        # y_train = to_categorical(y_train)
        # y_test = to_categorical(y_test)
        #
        # print(y_train[0])
        # endregion

        # Create model
        base_model = Sequential()

        # Add different model structures and test them separately
        if current_model == 1:
            run_model_1(base_model, x_train, y_train, x_test, y_test)
        elif current_model == 2:
            run_model_2(base_model, x_train, y_train, x_test, y_test)
        elif current_model == 3:
            run_model_3(base_model, x_train, y_train, x_test, y_test)


def get_data_from_video(video_name, input_name, compression_percentage, model_code):
    data_loader = DataLoader(video_name, input_name, False, 0, compression_percentage)
    video_data, input_data = data_loader.get_data()

    set_global_variables(model_code)

    video_data = normalise_video_data(video_data)
    video_data = np.asarray(clean_data(video_data, current_data_format, current_chunk_size))
    input_data = np.asarray(clean_data(input_data, current_data_format, current_chunk_size))

    video_data, input_data = transform_data_to_equal_length(video_data, input_data)

    input_data = categorise_input_data(input_data, current_input_format)
    return video_data, input_data


def convert_data_to_train_test_batches(video_data, input_data, ratio=0.8):
    train_test_ratio = ratio

    indices = np.random.permutation(video_data.shape[0])
    train_indices, test_indices = indices[:int(len(indices) * train_test_ratio)], indices[
                                                                                  int(len(indices) * train_test_ratio):]
    x_train, x_test = video_data[train_indices, :], video_data[test_indices, :]
    y_train, y_test = input_data[train_indices, :], input_data[test_indices, :]
    return x_train, x_test, y_train, y_test


def load_model_from_file(model_code):
    model = load_model("../../networks/{}.h5".format(model_code))
    return model


input_keys_amount = 5
models_to_run = [1311, 2315, 3315, 1321, 2325, 3325, 1331, 2335, 3335]

train_networks(models_to_run)

# video_data, input_data = get_data_from_video("output_7.avi", "inputs_7.csv", 20, 1331)
# model = load_model_from_file("1331-20-08")

# for chunk in video_data:
# result = model.predict(video_data)

# show_video(video_data, input_data, result, 1000)

terminate_program(getframeinfo(currentframe()).lineno)

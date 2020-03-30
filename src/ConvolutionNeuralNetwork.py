from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Conv3D
from keras.optimizers import Adam, SGD
from keras.constraints import max_norm
import keras.backend as k_backend
from DataLoader import DataLoader
import cv2
import sys
import numpy as np
from inspect import currentframe, getframeinfo
import json
from os import path


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
            else:
                temp_categorised_input_data[i][chunk_number][0] = 0
            if "down" in data:
                temp_categorised_input_data[i][chunk_number][1] = 1
            else:
                temp_categorised_input_data[i][chunk_number][1] = 0
            if "left" in data:
                temp_categorised_input_data[i][chunk_number][2] = 1
            else:
                temp_categorised_input_data[i][chunk_number][2] = 0
            if "right" in data:
                temp_categorised_input_data[i][chunk_number][3] = 1
            else:
                temp_categorised_input_data[i][chunk_number][3] = 0
            if "z" in data:
                temp_categorised_input_data[i][chunk_number][4] = 1
            else:
                temp_categorised_input_data[i][chunk_number][4] = 0
        categorised_input_data[i] = temp_categorised_input_data[i].flatten().astype(int)
    return categorised_input_data
# endregion


# region Create CNN models
def create_model_1(base_model):
    # region Model 1
    model_1 = build_model_1(base_model)

    # Compile model using accuracy to measure model performance
    # model_1.compile(optimizer=Adam(lr=current_learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
    model_1.compile(optimizer=SGD(lr=current_learning_rate, clipnorm=1), loss="mean_squared_error", metrics=["accuracy"])

    # Save the model
    save_model(model_1)
    # endregion


def create_model_2(base_model):
    # region Model 2
    model_2 = build_model_2(base_model)

    # Compile model using accuracy to measure model performance
    model_2.compile(optimizer=Adam(lr=current_learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])

    # Save the model
    save_model(model_2)
    # endregion


def create_model_3(base_model):
    # region Model 3
    model_3 = build_model_3(base_model)

    # Compile model using accuracy to measure model performance
    model_3.compile(optimizer=Adam(lr=current_learning_rate), loss="mean_squared_error", metrics=["accuracy"])

    # Save the model
    save_model(model_3)
    # endregion


def create_model_4(base_model):
    # region Model 4
    model_4 = build_model_4(base_model)

    # Compile model using accuracy to measure model performance
    model_4.compile(optimizer=Adam(lr=current_learning_rate), loss="mean_squared_error", metrics=["accuracy"])

    # Save the model
    save_model(model_4)
    # endregion


def train_model(model, x_train, x_test, y_train, y_test, video_data=None):
    weights, biases = model.layers[0].get_weights()
    # print("weights")
    # print(weights)
    # print("biases")
    # print(biases)
    result = model.predict(x_test)
    print(result)
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=current_epochs)

    weights, biases = model.layers[0].get_weights()
    # print("weights")
    # print(weights)
    # print("biases")
    # print(biases)

    save_model(model, history)
# endregion


# region Build CNN models
def build_model_1(model):
    model.add(Dense(current_layer_size, activation="softmax", input_shape=(28, 28, 1), bias_constraint=max_norm(3)))
    model.add(Flatten())
    model.add(Dense(input_keys_amount * current_input_size_modifier * 2, activation=current_activation_function, bias_constraint=max_norm(3)))
    return model


def build_model_2(model):
    model.add(Conv2D(current_layer_size, kernel_size=current_kernel_size, activation="relu", input_shape=(95, 103, 1), data_format="channels_last"))
    model.add(Flatten())
    model.add(Dense(input_keys_amount * current_input_size_modifier, activation=current_activation_function))
    return model


def build_model_3(model):
    model.add(Conv2D(current_layer_size, kernel_size=current_kernel_size, activation="relu", input_shape=(95, 103, 1)))
    model.add(Conv2D(int(current_layer_size / 2), kernel_size=3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(input_keys_amount * current_input_size_modifier, activation=current_activation_function))

    # model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28, 28, 1)))
    # model.add(Conv2D(32, kernel_size=3, activation="relu"))
    # model.add(Flatten())
    # model.add(Dense(10, activation="softmax"))
    return model


def build_model_4(model):
    model.add(Dense(current_layer_size, activation="relu", input_shape=(current_chunk_size, 95, 103)))
    model.add(Dense(current_layer_size))
    model.add(Dense(current_layer_size))
    model.add(Flatten())
    model.add(Dense(input_keys_amount * current_input_size_modifier, activation=current_activation_function))
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
    global current_layer_size
    global current_kernel_size
    global current_activation_function
    global current_epochs
    global model_file_name
    global current_learning_rate

    model_file_name = model_code
    current_model_code = model_code.split("_")
    for i, code in enumerate(current_model_code):
        current_model_code[i] = code.split("-")

    current_model = int(current_model_code[0][0])
    current_layer_size = int(current_model_code[0][1])
    current_kernel_size = int(current_model_code[0][2])
    current_activation_function = get_activation_function(current_model_code[0][3])
    current_epochs = int(current_model_code[0][4])
    current_learning_rate = float(current_model_code[0][5])
    current_data_format = int(current_model_code[1][0])
    current_input_format = int(current_model_code[1][1])
    current_chunk_size = int(current_model_code[1][2])

    current_input_size_modifier = 1 if current_input_format == 3 else current_chunk_size


def get_chunk_size(value):
    if value == 1:
        return 1
    elif value == 2 or value == 3:
        return 3


def split(word):
    return [char for char in str(word)]


def get_activation_function(code):
    code = int(code)
    if code == 0:
        return "elu"
    elif code == 1:
        return "relu"
    elif code == 2:
        return "sigmoid"
    elif code == 3:
        return "softmax"
    elif code == 4:
        return "tanh"
    print("Error")
    return "elu"


def save_model(model, history=None):
    model.save("../../networks/accuracy-test/{}.h5".format(model_file_name))

    if history is not None:
        with open("../../networks/accuracy-test/{}_history.json".format(model_file_name), "w+") as file:
            json.dump(str(history.history), file)

# endregion


# region DEBUG
def show_video(video_data, input_data, result=None, playback_speed=10):
    for i, chunk in enumerate(video_data):
        for j, frame in enumerate(chunk):
            # print("Recorded inputs: {}".format(input_data[i]))
            print("Recorded inputs: {}".format(np.round(input_data[i], 2)))
            if result is not None:
                # print("Model prediction: {}".format(result[i]))
                print("Model prediction: {}".format(np.round(result[i], 2)))
            cv2.imshow("Test", frame)

            if cv2.waitKey(playback_speed) & 0xFF == ord("q"):
                pass
    cv2.destroyAllWindows()
    terminate_program(getframeinfo(currentframe()).lineno, "Debug video ended")


def terminate_program(line_number=0, message="Program terminated (DEBUG)"):
    sys.exit(message + "\nLine: {}".format(line_number))
# endregion DEBUG


def train_networks(models_to_run, videos_to_use):
    for model_code in models_to_run:
        print(model_code)
        set_global_variables(model_code)

        total_video_data = []
        total_input_data = []

        for index in videos_to_use:
            video_data, input_data = get_data_from_video("output_{}.avi".format(index), "inputs_{}.csv".format(index), 20, model_code)

            for i, video in enumerate(video_data):
                total_video_data.append(video_data[i])

            for i, input in enumerate(input_data):
                total_input_data.append(input_data[i])

        total_video_data = np.array(total_video_data)
        total_input_data = np.array(total_input_data)

        print("Model: {} - Layer Size: {} - Kernel Size: {} - Activation: {} - "
              "Epochs: {} - Learning Rate: {} - Data Format: {} - Input Format: {} - Chunk Size: {}".format(
                current_model, current_layer_size, current_kernel_size, current_activation_function, current_epochs,
                current_learning_rate, current_data_format, current_input_format, current_chunk_size))

        # show_video(video_data, input_data)
        # terminate_program(getframeinfo(currentframe()).lineno)

        x_train, x_test, y_train, y_test = convert_data_to_train_test_batches(total_video_data, total_input_data, 0.9)

        x_train = x_train.reshape(len(x_train), 95, 103, 1)
        x_test = x_test.reshape(len(x_test), 95, 103, 1)

        print(x_train.shape)
        print(x_test.shape)

        # # region Mnist test
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
        # # endregion

        model = load_model_from_file(model_file_name)

        if model is None:
            # Create model
            base_model = Sequential()

            # Add different model structures and test them separately
            if current_model == 1:
                create_model_1(base_model)
            elif current_model == 2:
                create_model_2(base_model)
            elif current_model == 3:
                create_model_3(base_model)
            elif current_model == 4:
                create_model_4(base_model)
            print("Created new network")
        train_model(load_model_from_file(model_file_name), x_train, x_test, y_train, y_test, total_video_data)


def get_data_from_video(video_name, input_name, compression_percentage, model_code):
    data_loader = DataLoader(video_name, input_name, False, 0, compression_percentage, greyscale=True)
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
    if path.exists("../../networks/accuracy-test/{}.h5".format(model_code)):
        model = load_model("../../networks/accuracy-test/{}.h5".format(model_code))
    else:
        model = None
    return model


input_keys_amount = 5

possible_models = [2, 3]
possible_layer_sizes = [32, 64, 128]
possible_kernel_sizes = [3]
possible_activation_functions = [0, 1, 2, 3, 4]
possible_epochs = [1, 3]
possible_learning_rates = [0.001, 0.0001, 0.00001]
possible_data_formats = ["1-3-1", "2-3-3", "2-3-5", "3-3-3", "3-3-5"]

models_to_run = ["3-64-3-2-1-0.001_1-3-1"]

# for model in possible_models:
#     for layer_size in possible_layer_sizes:
#         for kernel_size in possible_kernel_sizes:
#             for activation_function in possible_activation_functions:
#                 for epoch in possible_epochs:
#                     for learning_rate in possible_learning_rates:
#                         for data_format in possible_data_formats:
#                             if int(data_format[4]) == 1:
#                                 kernel_size = 1
#                             models_to_run.append("{}-{}-{}-{}-{}-{:f}_{}".format(
#                                 model, layer_size, kernel_size, activation_function, epoch, learning_rate, data_format))

# region Old models to run
# models_to_run = ["2-64-3-0-1-0.001_2-3-5", "2-64-3-0-1-0.0001_2-3-5", "2-64-3-0-1-0.00001_2-3-5",
#                  "2-32-3-0-1-0.001_2-3-5", "2-32-3-0-1-0.0001_2-3-5", "2-32-3-0-1-0.00001_2-3-5",
#                  "2-128-3-0-1-0.001_2-3-5", "2-128-3-0-1-0.0001_2-3-5", "2-128-3-0-1-0.00001_2-3-5",
#                  "2-64-3-0-3-0.001_2-3-5", "2-64-3-0-3-0.0001_2-3-5", "2-64-3-0-3-0.00001_2-3-5",
#                  "2-32-3-0-3-0.001_2-3-5", "2-32-3-0-3-0.0001_2-3-5", "2-32-3-0-3-0.00001_2-3-5",
#                  "2-128-3-0-3-0.001_2-3-5", "2-128-3-0-3-0.0001_2-3-5", "2-128-3-0-3-0.00001_2-3-5",
#
#                  "2-64-3-1-1-0.001_2-3-5", "2-64-3-1-1-0.0001_2-3-5", "2-64-3-1-1-0.00001_2-3-5",
#                  "2-32-3-1-1-0.001_2-3-5", "2-32-3-1-1-0.0001_2-3-5", "2-32-3-1-1-0.00001_2-3-5",
#                  "2-128-3-1-1-0.001_2-3-5", "2-128-3-1-1-0.0001_2-3-5", "2-128-3-1-1-0.00001_2-3-5",
#                  "2-64-3-1-3-0.001_2-3-5", "2-64-3-1-3-0.0001_2-3-5", "2-64-3-1-3-0.00001_2-3-5",
#                  "2-32-3-1-3-0.001_2-3-5", "2-32-3-1-3-0.0001_2-3-5", "2-32-3-1-3-0.00001_2-3-5",
#                  "2-128-3-1-3-0.001_2-3-5", "2-128-3-1-3-0.0001_2-3-5", "2-128-3-1-3-0.00001_2-3-5",
#
#                  "2-64-3-2-1-0.001_2-3-5", "2-64-3-2-1-0.0001_2-3-5", "2-64-3-2-1-0.00001_2-3-5",
#                  "2-32-3-2-1-0.001_2-3-5", "2-32-3-2-1-0.0001_2-3-5", "2-32-3-2-1-0.00001_2-3-5",
#                  "2-128-3-2-1-0.001_2-3-5", "2-128-3-2-1-0.0001_2-3-5", "2-128-3-2-1-0.00001_2-3-5",
#                  "2-64-3-2-3-0.001_2-3-5", "2-64-3-2-3-0.0001_2-3-5", "2-64-3-2-3-0.00001_2-3-5",
#                  "2-32-3-2-3-0.001_2-3-5", "2-32-3-2-3-0.0001_2-3-5", "2-32-3-2-3-0.00001_2-3-5",
#                  "2-128-3-2-3-0.001_2-3-5", "2-128-3-2-3-0.0001_2-3-5", "2-128-3-2-3-0.00001_2-3-5",
#
#                  "2-64-3-3-1-0.001_2-3-5", "2-64-3-3-1-0.0001_2-3-5", "2-64-3-3-1-0.00001_2-3-5",
#                  "2-32-3-3-1-0.001_2-3-5", "2-32-3-3-1-0.0001_2-3-5", "2-32-3-3-1-0.00001_2-3-5",
#                  "2-128-3-3-1-0.001_2-3-5", "2-128-3-3-1-0.0001_2-3-5", "2-128-3-3-1-0.00001_2-3-5",
#                  "2-64-3-3-3-0.001_2-3-5", "2-64-3-3-3-0.0001_2-3-5", "2-64-3-3-3-0.00001_2-3-5",
#                  "2-32-3-3-3-0.001_2-3-5", "2-32-3-3-3-0.0001_2-3-5", "2-32-3-3-3-0.00001_2-3-5",
#                  "2-128-3-3-3-0.001_2-3-5", "2-128-3-3-3-0.0001_2-3-5", "2-128-3-3-3-0.00001_2-3-5",
#
#                  "2-64-3-4-1-0.001_2-3-5", "2-64-3-4-1-0.0001_2-3-5", "2-64-3-4-1-0.00001_2-3-5",
#                  "2-32-3-4-1-0.001_2-3-5", "2-32-3-4-1-0.0001_2-3-5", "2-32-3-4-1-0.00001_2-3-5",
#                  "2-128-3-4-1-0.001_2-3-5", "2-128-3-4-1-0.0001_2-3-5", "2-128-3-4-1-0.00001_2-3-5",
#                  "2-64-3-4-3-0.001_2-3-5", "2-64-3-4-3-0.0001_2-3-5", "2-64-3-4-3-0.00001_2-3-5",
#                  "2-32-3-4-3-0.001_2-3-5", "2-32-3-4-3-0.0001_2-3-5", "2-32-3-4-3-0.00001_2-3-5",
#                  "2-128-3-4-3-0.001_2-3-5", "2-128-3-4-3-0.0001_2-3-5", "2-128-3-4-3-0.00001_2-3-5"
#                  ]
# "1-64-0-0-1_1-3-1", "1-32-0-0-1_1-3-1", "1-128-0-0-1_1-3-1",

# "1-64-0-1-1_1-3-1", "1-32-0-1-1_1-3-1", "1-128-0-1-1_1-3-1",
#                  "1-64-0-2-1_1-3-1", "1-32-0-2-1_1-3-1", "1-128-0-2-1_1-3-1",
#                  "1-64-0-4-1_1-3-1", "1-32-0-4-1_1-3-1", "1-128-0-4-1_1-3-1",
# endregion

video_list = [7, 8, 9, 10, 11]
# video_list = [7, 8, 9, 10]
# , 1311, 2315, 3315, 1321, 2325, 3325, 1331, 2335, 3335

train_networks(models_to_run, video_list)

video_data, input_data = get_data_from_video("output_11.avi", "inputs_11.csv", 20, "3-64-3-2-1-0.001_1-3-1")
video_data_reshape = video_data.reshape(len(video_data), 95, 103, 1)
model = load_model_from_file("3-64-3-2-1-0.001_1-3-1")

result = model.predict(video_data_reshape)

print(result)
print(model.summary())

show_video(video_data, input_data, result, 200)

terminate_program(getframeinfo(currentframe()).lineno)

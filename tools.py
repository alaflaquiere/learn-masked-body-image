#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import glob
import json
import matplotlib.pyplot as plt
import tensorflow as tf


class Normalizer:

    def __init__(self, low=0, high=1, min_data=None, max_data=None):
        if low < high:
            self.low, self.high = low, high
        elif high < low:
            print("Warning: lower limit greater than higher limit - inverting the two")
            self.low, self.high = high, low
        else:
            print("Error: lower limit equal to higher limit")
            self.low, self.high = low, high
        self.min_data = min_data
        self.max_data = max_data

    def fit(self, x):
        if (self.min_data is not None) or (self.max_data is not None):
            print("Error: trying to overwrite the class attributes - forbidden operation")
            return None
        self.min_data = np.min(x, axis=0)
        self.max_data = np.max(x, axis=0)
        return self.min_data, self.max_data

    def transform(self, x):
        if (self.min_data is None) or (self.max_data is None):
            print("Error: fit to data before using transform")
            return None
        x_n = (x - self.min_data).astype(np.float)
        x_n = np.divide(x_n, self.max_data - self.min_data, out=np.zeros_like(x_n), where=self.max_data - self.min_data != 0)
        x_n = x_n * (self.high - self.low) + self.low
        return x_n

    def fit_transform(self, x):
        self.fit(x)
        x_n = self.transform(x)
        return x_n

    def reconstruct(self, y):
        if (self.min_data is None) or (self.max_data is None):
            print("Error: fit to data before using reconstruct")
            return None
        x = (y - self.low) / (self.high - self.low)
        x = x * (self.max_data - self.min_data)
        x = x + self.min_data
        return x


def create_checkerboard(height, width):
    square_light_gray = np.full((5, 5, 3), 250)
    square_dark_gray = np.full((5, 5, 3), 200)
    checker_square = np.vstack((np.hstack((square_dark_gray, square_light_gray)), np.hstack((square_light_gray, square_dark_gray))))
    checkerboard = np.tile(checker_square, (int(np.ceil(height / checker_square.shape[0])), int(np.ceil(width / checker_square.shape[1])), 1))
    return checkerboard[:height, :width, :]


def load_data(dir_dataset):
    """
    Load sensorimotor data.

    Parameters:
        dir_dataset - dataset directory
    """

    # check directories
    if not os.path.exists(dir_dataset):
        print("Error: the dataset directory {} doesn't exist.".format(dir_dataset))
        return
    # check the content of the directory
    images_list = glob.glob(dir_dataset + "/*.png")
    if len(images_list) == 0:
        print("Error: the directory {} doesn't contain any png image.".format(dir_dataset))
        return
    if not os.path.exists(dir_dataset + "/positions.txt"):
        print("Error: the directory {} doesn't contain a positions.txt file.".format(dir_dataset))
        return

    # load the motor data
    with open(dir_dataset + "/positions.txt", "r") as file:
        m = np.array(json.load(file))

    # load the sensory data
    for i, file in enumerate(images_list):
        img = plt.imread(file)
        if i == 0:
            s = np.full((len(images_list), img.shape[0], img.shape[1], 3), np.nan)
        s[i, :, :, :] = img

    # get dataset parameters
    number_samples, height, width, number_channels = s.shape
    number_joints = m.shape[1]

    # check the data compatibility
    temp = m.shape[0]
    if not number_samples == temp:
        print("Error: incompatible number of motor_input configurations and images ({} != {})".format(temp, number_samples))
        return
    print("loaded data: {} samples, {} joints, {}x{}x{} images".format(number_samples, number_joints, height, width, number_channels))

    return m, s, number_samples, height, width, number_channels, number_joints


def load_network(dir_model):
    """
    Load a network and return useful placeholders data.

    Parameters:
        dir_dataset - dataset directory
    """

    # check model directory
    if not os.path.exists(dir_model) or not os.path.exists(dir_model + "/network.ckpt.meta"):
        print("Error: the directory {} doesn't exist or doesn't contain a network.ckpt.meta file.".format(dir_model))
        return

    # reload the graph
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(dir_model + "/network.ckpt.meta")
    graph = tf.get_default_graph()

    # recover the input and outputs
    motor_input = graph.get_tensor_by_name("motor_input:0")
    predicted_image = graph.get_tensor_by_name("image_branch/predicted_image/Relu:0")
    predicted_error = graph.get_tensor_by_name("error_branch/predicted_error/Relu:0")

    return saver, motor_input, predicted_image, predicted_error


def _load_data(dir_dataset):
    # DEPRECATED

    print("loading the data...")

    if os.path.exists(dir_dataset + "/positions.txt") and not glob.glob(dir_dataset + "/*.png") == []:
        print("new data format")
        data_format = "new"
    elif os.path.exists(dir_dataset + "/positions.json") and os.path.exists(dir_dataset + "/images.json"):
        print("old data format")
        data_format = "old"
    else:
        print("Error: incorrect path to dataset")

    if data_format is "new":

        # load the motor data
        with open(dir_dataset + "/positions.txt", "r") as file:
            m = np.array(json.load(file))

        # load the sensory data
        files_list = sorted(glob.glob(dir_dataset + "/*.png"))
        for i, file in enumerate(files_list):
            img = plt.imread(file)
            if i == 0:
                s = np.full((len(files_list), img.shape[0], img.shape[1], 3), np.nan)
            s[i, :, :, :] = img

    elif data_format is "old":

        # load the motor data
        with open(dir_dataset + "/positions.json", "r") as file:
            m = np.array(json.load(file))

        # load the sensory data
        with open(dir_dataset + "/images.json", "r") as file:
            s = np.array(json.load(file))

    return m, s

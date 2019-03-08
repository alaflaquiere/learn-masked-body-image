#!/usr/bin/env python
# coding: utf-8

import os
import glob
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import json
from tools import Normalizer, create_checkerboard, load_data
import matplotlib.pyplot as plt
plt.ion()


def create_network(n_joints, h, w):

    # create placeholders
    joint = tf.placeholder(dtype=tf.float32, shape=[None, n_joints], name="joint_config")
    gt_image = tf.placeholder(dtype=tf.float32, shape=[None, h, w, 3], name="gt_image")

    # dense mapping to larger layers
    out = tf.layers.dense(inputs=joint, units=8 * 8 * 3, activation=tf.nn.selu)
    out = tf.layers.dense(inputs=out, units=round(h/5) * round(w/5) * 3, activation=tf.nn.selu)

    # reshaping
    out = tf.reshape(out, shape=[-1, round(h/5), round(w/5), 3])

    # branch 1: image - deconvolution is done by upsampling + convolution - upsampling with +2 to compensate for the valid padding
    image = tf.image.resize_images(out, size=(round(h/4) + 2, round(w/4) + 2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
    image = tf.layers.conv2d(inputs=image, filters=32, kernel_size=(3, 3), padding='valid', activation=tf.nn.selu)
    # 
    image = tf.image.resize_images(image, size=(round(h/2) + 2, round(w/2) + 2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
    image = tf.layers.conv2d(inputs=image, filters=32, kernel_size=(3, 3), padding='valid', activation=tf.nn.selu)
    # 
    image = tf.image.resize_images(image, size=(h + 8, w + 8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
    image = tf.layers.conv2d(inputs=image, filters=32, kernel_size=(3, 3), padding='valid', activation=tf.nn.selu)
    #
    # convolutions + reducing the number of filters to 3 channels
    image = tf.layers.conv2d(inputs=image, filters=16, kernel_size=(3, 3), padding="valid", activation=tf.nn.selu)
    image = tf.layers.conv2d(inputs=image, filters=8, kernel_size=(3, 3), padding="valid", activation=tf.nn.selu)
    image = tf.layers.conv2d(inputs=image, filters=3, kernel_size=(3, 3), padding="valid", activation=tf.nn.relu, name="image")

    # branch 2 - prediction error
    mask = tf.image.resize_images(out, size=(round(h / 4) + 2, round(w / 4) + 2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
    mask = tf.layers.conv2d(inputs=mask, filters=32, kernel_size=(3, 3), padding='valid', activation=tf.nn.selu)
    # 
    mask = tf.image.resize_images(mask, size=(round(h / 2) + 2, round(w / 2) + 2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
    mask = tf.layers.conv2d(inputs=mask, filters=32, kernel_size=(3, 3), padding='valid', activation=tf.nn.selu)
    # 
    mask = tf.image.resize_images(mask, size=(h + 8, w + 8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
    mask = tf.layers.conv2d(inputs=mask, filters=32, kernel_size=(3, 3), padding='valid', activation=tf.nn.selu)
    #
    # convolutions + reducing the number of filters to 3 channels
    mask = tf.layers.conv2d(inputs=mask, filters=16, kernel_size=(3, 3), padding="valid", activation=tf.nn.selu)
    mask = tf.layers.conv2d(inputs=mask, filters=8, kernel_size=(3, 3), padding="valid", activation=tf.nn.selu)
    mask = tf.layers.conv2d(inputs=mask, filters=3, kernel_size=(3, 3), padding="valid", activation=tf.nn.relu, name="mask")

    return joint, gt_image, image, mask


class Display:

    def __init__(self, height, width):
        self.fig, self.ax1, self.ax2, self.ax3, self.ax4, self.ax5 = self.open_figure()
        self.checkerboard = create_checkerboard(height, width)

    @staticmethod
    def open_figure():
        fig = plt.figure(1, figsize=(14, 8))
        ax1 = fig.add_subplot(231)
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(234)
        ax4 = fig.add_subplot(235)
        ax5 = fig.add_subplot(133)
        return fig, ax1, ax2, ax3, ax4, ax5

    def plot(self, gt_image, pred_image, pred_error, mask):

        if not plt.fignum_exists(1):
            self.open_figure()

        self.ax1.cla()
        self.ax1.set_title("ground-truth image")
        self.ax1.imshow(gt_image)
        self.ax1.axis("off")

        self.ax2.cla()
        self.ax2.set_title("predicted image")
        self.ax2.imshow(pred_image)
        self.ax2.axis("off")

        self.ax3.cla()
        self.ax3.set_title("predicted prediction error")
        self.ax3.imshow(pred_error)
        self.ax3.axis("off")

        self.ax4.cla()
        self.ax4.set_title('mask')
        self.ax4.imshow(mask)
        self.ax4.axis("off")

        self.ax5.cla()
        masked_image = np.dstack((pred_image, np.mean(mask, axis=2)))
        self.ax5.set_title('masked prediction')
        self.ax5.imshow(self.checkerboard)
        self.ax5.imshow(masked_image)
        self.ax5.axis("off")

        plt.show(False)
        plt.pause(1e-8)

    def save(self, path, epoch):
        self.fig.savefig(path + "/epoch_{:06d}.png".format(epoch))


def train_network(dir_dataset="dataset/generated/combined", dir_model="model/trained", number_epochs=int(5e4), batch_size=100, save_progress=True):

    # check directories
    if not os.path.exists(dir_dataset):
        print("Error: incorrect path to the dataset")
        return

    if os.path.exists(dir_model):
        ans = input(" ".join(["> The folder", dir_model, "already exists; do you want to overwrite its content? [y,n]: "]))
        if ans is not "y":
            print("exiting the program")
            return

    if not os.path.exists(dir_model):
        os.makedirs(dir_model)

    if save_progress:
        dir_progress = dir_model + "/progress"
        if not os.path.exists(dir_progress):
            os.makedirs(dir_progress)

    # print("loading the data...")
    # # load the motor data
    # with open(dir_dataset + "/positions.txt", "r") as file:
    #     m = np.array(json.load(file))
    # number_joints = m.shape[1]
    # # load the sensory data
    # files_list = sorted(glob.glob(dir_dataset + "/*.png"))
    # image = plt.imread(files_list[0])
    # height, width = image.shape[0], image.shape[1]
    # s = np.full((len(files_list), height, width, 3), np.nan)
    # for i, file in enumerate(files_list):
    #     s[i, :, :, :] = plt.imread(file)
    m, s = load_data(dir_dataset)
    number_samples, height, width, number_channels = s.shape
    number_joints = m.shape[1]

    # check the data compatibility
    temp = m.shape[0]
    if not number_samples == temp:
        print("Error: incompatible number of joint configurations and images ({} != {})".format(temp, number_samples))
        return
    print("loaded data: {} samples, {} joints, {}x{}x{} images".format(number_samples, number_joints, height, width, number_channels))

    # normalize the joint configuration in [-1, 1]
    m_normalizer = Normalizer(low=-1, high=1)
    m = m_normalizer.fit_transform(m)

    # normalize the pixel channels in [0, 1] (doesn't change anything in this case, as plt.imread already outputs values in [0, 1]
    s_normalizer = Normalizer(low=0, high=1, min_data=0, max_data=1)
    s = s_normalizer.transform(s)

    # create the network
    tf.reset_default_graph()
    joint, gt_image, net_image, net_mask = create_network(number_joints, height, width)

    # define the loss
    errors_image = tf.abs(tf.subtract(net_image, gt_image))
    loss_reconstruction = tf.reduce_mean(errors_image)
    error_mask = tf.abs(tf.subtract(net_mask, errors_image))
    loss_mask = tf.reduce_mean(error_mask)
    loss = loss_reconstruction + loss_mask

    # define the optimizer
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.polynomial_decay(1e-3, global_step, number_epochs, 1e-5, power=1)
    # define the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    minimize_op = optimizer.minimize(loss, global_step=global_step)

    # define a instance to display the network output
    output_display = Display(height, width)

    # train the network
    print("training the network...")
    with tf.Session() as sess:

        # initialize the network
        sess.run(tf.global_variables_initializer())

        for epoch in range(number_epochs):

            # draw batch indexes
            indexes = np.random.choice(number_samples, batch_size, replace=True)

            # minimize the loss
            curr_loss, _, curr_lr = sess.run([loss, minimize_op, learning_rate], feed_dict={joint: m[indexes, :], gt_image: s[indexes, :, :]})

            if (epoch % max(1, np.round(number_epochs/100)) == 0) or (epoch == number_epochs - 1):

                print("epoch: {} ({:3.0f}%), learning rate: {:.4e}, loss: {:.4e}".format(epoch, epoch/number_epochs*100, curr_lr, curr_loss))

                # visualize one output
                curr_mask, curr_image = sess.run([net_mask, net_image], feed_dict={joint: m[[indexes[0]], :]})
                curr_image = s_normalizer.reconstruct(curr_image)  # identity mapping in this case, as the pixel values are already in [0, 1]
                binary_mask = (curr_mask[0] < 0.056).astype(float)  # todo: fit a GMM on the fly to have a dynamic threshold instead of a fixed value
                output_display.plot(s[indexes[0]], curr_image[0], curr_mask[0], binary_mask)

                # save the figure
                if save_progress:
                    output_display.save(dir_progress, epoch)

                # save the network
                saver = tf.train.Saver()
                saver.save(sess, dir_model + "/network.ckpt")  # add the argument global_step=global_step do avoid overwriting the previous model

    print("done")
    plt.show(block=True)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-dd", "--dir_dataset", dest="dir_dataset", help="dataset directory", default="dataset/generated/combined")
    parser.add_argument("-dm", "--dir_model", dest="dir_model", help="directory in which to save the model", default="model/trained")
    parser.add_argument("-n", "--n_epochs", dest="number_epochs", help="number of mini-batch epochs", type=int, default=5e4)
    parser.add_argument("-b", "--batch_size", dest="batch_size", help="mini-batch size", type=int, default=100)
    parser.add_argument("-s", "--save_progress", dest="save_progress", help="flag to save the training visualization", type=bool, default=True)

    args = parser.parse_args()
    dir_dataset = args.dir_dataset
    dir_model = args.dir_model
    number_epochs = int(args.number_epochs)
    batch_size = args.batch_size
    save_progress = args.save_progress

    train_network(dir_dataset=dir_dataset, dir_model=dir_model, number_epochs=number_epochs, batch_size=batch_size, save_progress=save_progress)

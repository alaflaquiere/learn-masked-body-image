#!/usr/bin/env python
# coding: utf-8

import os
import glob
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
from tools import Normalizer, create_checkerboard, load_data
import matplotlib.pyplot as plt
plt.ion()


class SensoriMotorPredictionNetwork:

        def __init__(self, n_joints, height, width, n_filter):
            self.n_joints = n_joints
            self.height = height
            self.width = width
            self.n_filter = n_filter
            self.motor_input, self.gt_image, self.predicted_image, self.predicted_error, self.weight_error_loss, self.loss =\
                self.create_network(self.n_joints, self.height, self.width, n_filter)
            self.m_normalizer = Normalizer(low=-1, high=1)
            self.s_normalizer = Normalizer(low=0, high=1, min_data=0, max_data=1) # equal identity here as pixels are already in [0,1]
            self.saver = tf.train.Saver()
            self.fig = plt.figure(1, figsize=(14, 8))

        @staticmethod
        def create_network(n_joints, h, w, n_filter=32):
            """
            Create the network for sensorimotor prediction.
            Given an input motor configuration, the network outputs a predictive image and predicted prediction error.

            Parameters:
                n_joints - dimension of the motor states
                h - height of the output image
                w - width of the output image
                n_filter - maximal number of convolution filters
            """
            # todo: test padding="same" for the final convolutional layers
            # todo: test with batch normalization

            # reset the default graph
            tf.reset_default_graph()

            # create placeholders
            motor_input = tf.placeholder(dtype=tf.float32, shape=[None, n_joints], name="motor_input")
            gt_image = tf.placeholder(dtype=tf.float32, shape=[None, h, w, 3], name="gt_image")
            weight_error_loss = tf.placeholder(dtype=tf.float32, shape=[], name="weight_error_loss")

            # dense mapping to larger layers
            with tf.name_scope("dense_expand") as scope:
                out = tf.layers.dense(inputs=motor_input, units=8 * 8 * 3, activation=tf.nn.selu, name="layer1")
                out = tf.layers.dense(inputs=out, units=round(h/5) * round(w/5) * 3, activation=tf.nn.selu, name="layer2")

            # reshaping
            out = tf.reshape(out, shape=[-1, round(h/5), round(w/5), 3], name="reshaping")

            # branch 1: image - deconvolution is done by upsampling + convolution - upsampling with +2 to compensate for the valid padding
            with tf.variable_scope("image_branch") as scope:
                img = tf.image.resize_images(out, size=(round(h/4) + 2, round(w/4) + 2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
                img = tf.layers.conv2d(inputs=img, filters=n_filter, kernel_size=(3, 3), padding='valid', activation=tf.nn.selu, name="deconv1")
                #
                img = tf.image.resize_images(img, size=(round(h/2) + 2, round(w/2) + 2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
                img = tf.layers.conv2d(inputs=img, filters=n_filter, kernel_size=(3, 3), padding='valid', activation=tf.nn.selu, name="deconv2")
                #
                img = tf.image.resize_images(img, size=(h + 8, w + 8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
                img = tf.layers.conv2d(inputs=img, filters=n_filter, kernel_size=(3, 3), padding='valid', activation=tf.nn.selu, name="deconv3")
                #
                # convolutions + reducing the number of filters to 3 channels
                img = tf.layers.conv2d(inputs=img, filters=n_filter/2, kernel_size=(3, 3), padding="valid", activation=tf.nn.selu, name="conv1")
                img = tf.layers.conv2d(inputs=img, filters=n_filter/4, kernel_size=(3, 3), padding="valid", activation=tf.nn.selu, name="conv2")
                img = tf.layers.conv2d(inputs=img, filters=3, kernel_size=(3, 3), padding="valid", activation=tf.nn.relu, name="predicted_image")

            # branch 2 - prediction error
            with tf.variable_scope("error_branch") as scope:
                err = tf.image.resize_images(out, size=(round(h/4) + 2, round(w/4) + 2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
                err = tf.layers.conv2d(inputs=err, filters=n_filter, kernel_size=(3, 3), padding='valid', activation=tf.nn.selu, name="deconv1")
                #
                err = tf.image.resize_images(err, size=(round(h/2) + 2, round(w/2) + 2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
                err = tf.layers.conv2d(inputs=err, filters=n_filter, kernel_size=(3, 3), padding='valid', activation=tf.nn.selu, name="deconv2")
                #
                err = tf.image.resize_images(err, size=(h + 8, w + 8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
                err = tf.layers.conv2d(inputs=err, filters=n_filter, kernel_size=(3, 3), padding='valid', activation=tf.nn.selu, name="deconv3")
                #
                # convolutions + reducing the number of filters to 3 channels
                err = tf.layers.conv2d(inputs=err, filters=n_filter/2, kernel_size=(3, 3), padding="valid", activation=tf.nn.selu, name="conv1")
                err = tf.layers.conv2d(inputs=err, filters=n_filter/4, kernel_size=(3, 3), padding="valid", activation=tf.nn.selu, name="conv2")
                err = tf.layers.conv2d(inputs=err, filters=3, kernel_size=(3, 3), padding="valid", activation=tf.nn.relu, name="predicted_error")

            # define the loss
            with tf.name_scope("losses_computation") as scope:
                errors_image = tf.abs(tf.subtract(img, gt_image), name="errors_images")
                loss_reconstruction = tf.reduce_mean(errors_image, name="loss_reconstruction")
                errors_mask = tf.abs(tf.subtract(err, errors_image), name="errors_mask")
                loss_mask = tf.reduce_mean(errors_mask, name="loss_error")
                loss_mask = tf.multiply(weight_error_loss, loss_mask, name="weighted_loss_error")
                loss = tf.add(loss_reconstruction, loss_mask, name="loss")

            return motor_input, gt_image, img, err, weight_error_loss, loss,

        def save_network(self):
            self.saver.save(tf.get_default_session(), dir_model + "/network.ckpt")  # add global_step=global_step to not overwrite the previous model

        def train(self, m, s, dir_model="model/trained", n_epochs=int(5e4), batch_size=100):
            """
            Train the network.
            The error-predition component of the loss is weighted with an weight increasing from 0 to 1 during the first half of training.

            Parameters:
                m - motor data
                s - sensor data (images)
                dir_model - directory where to save the model
                n_epochs - number of mini-batch iterations
                batch_size - mini-batch size
            """

            # check the model directory
            if os.path.exists(dir_model):
                ans = input("> The folder {} already exists; do you want to overwrite its content? [y,n]: ".format(dir_model))
                if ans is not "y":
                    print("exiting the program")
                    return

            # create directories if necessary
            if not os.path.exists(dir_model):
                os.makedirs(dir_model)
            dir_progress = dir_model + "/progress"
            if not os.path.exists(dir_progress):
                os.makedirs(dir_progress)

            # get the number of samples
            n_samples = s.shape[0]

            # normalize the motor_input configuration in [-1, 1]
            m = self.m_normalizer.fit_transform(m)

            # normalize the pixel channels in [0, 1] (doesn't change anything in this case, as plt.imread already outputs values in [0, 1]
            s = self.s_normalizer.transform(s)

            # define the optimizer
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.polynomial_decay(1e-3, global_step, n_epochs, 1e-5, power=1)
            # define the optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            minimize_op = optimizer.minimize(self.loss, global_step=global_step)

            # define the weighting of the error_loss - ramp up from 0 to 1 during the first half of training
            weight_err = tf.train.polynomial_decay(0., global_step, n_epochs//2, 1., power=1)

            # train the network
            print("training the network...")
            with tf.Session() as sess:

                # initialize the network
                sess.run(tf.global_variables_initializer())

                for epoch in range(n_epochs):

                    # draw batch indexes
                    indexes = np.random.choice(n_samples, batch_size, replace=True)

                    # minimize the loss
                    curr_weight = sess.run(weight_err)
                    curr_loss, _, curr_lr = sess.run([self.loss, minimize_op, learning_rate], feed_dict={self.motor_input: m[indexes, :],
                                                                                                         self.gt_image: s[indexes, :, :],
                                                                                                         self.weight_error_loss: curr_weight})

                    if (epoch % max(1, np.round(n_epochs/100)) == 0) or (epoch == n_epochs - 1):

                        print("epoch: {} ({:3.0f}%), learning rate: {:.4e}, error loss weight: {:.4e}, loss: {:.4e}".format(epoch, epoch/n_epochs*100,
                                                                                                                            curr_lr, curr_weight,
                                                                                                                            curr_loss))

                        # visualize one output
                        curr_image, curr_error = sess.run([self.predicted_image, self.predicted_error],
                                                          feed_dict={self.motor_input: m[[indexes[0]], :]})
                        curr_image = self.s_normalizer.reconstruct(curr_image)
                        binary_mask = (curr_error[0] < 0.056).astype(float)  # the htreshold value could be estimated on the fly with a GMM
                        self.display_figure(s[indexes[0]], curr_image[0], curr_error[0], binary_mask)

                        # save the visualization
                        self.save_figure(dir_progress, epoch)

                        # save the network
                        self.save_network()

            print("training finished.")

        def display_figure(self, gt_image, pred_image, pred_error, mask):
            """
            Display the output of the network for one input sample.

            Parameters:
                gt_image - ground truth image
                pred_image - predicted image
                pred_error - predicted prediction error
                mask - estimated mask
            """

            if not plt.fignum_exists(1):
                self.fig = plt.figure(1, figsize=(14, 8))

            # clean the figure
            plt.clf()
            ax1 = self.fig.add_subplot(231)
            ax2 = self.fig.add_subplot(232)
            ax3 = self.fig.add_subplot(234)
            ax4 = self.fig.add_subplot(235)
            ax5 = self.fig.add_subplot(133)

            checkerboard = create_checkerboard(pred_image.shape[0], pred_image.shape[1])

            ax1.cla()
            ax1.set_title("ground-truth image")
            ax1.imshow(gt_image)
            ax1.axis("off")

            ax2.cla()
            ax2.set_title("predicted image")
            ax2.imshow(pred_image)
            ax2.axis("off")

            ax3.cla()
            ax3.set_title("predicted prediction error")
            ax3.imshow(pred_error)
            ax3.axis("off")

            ax4.cla()
            ax4.set_title('mask')
            ax4.imshow(mask)
            ax4.axis("off")

            ax5.cla()
            masked_image = np.dstack((pred_image, np.mean(mask, axis=2)))
            ax5.set_title('masked prediction')
            ax5.imshow(checkerboard)
            ax5.imshow(masked_image)
            ax5.axis("off")

            plt.show(block=False)
            plt.pause(1e-8)

        def save_figure(self, path, epoch):
            self.fig.savefig(path + "/epoch_{:06d}.png".format(epoch))


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-dd", "--dir_dataset", dest="dir_dataset", help="dataset directory", default=".dataset/generated/combined")
    parser.add_argument("-dm", "--dir_model", dest="dir_model", help="directory in which to save the model", default=".model/trained")
    parser.add_argument("-n", "--n_epochs", dest="number_epochs", help="number of mini-batch epochs", type=int, default=5e4)
    parser.add_argument("-b", "--batch_size", dest="batch_size", help="mini-batch size", type=int, default=100)
    parser.add_argument("-nf", "--n_filters", dest="n_filters", help="maximal number of convolutional filters", type=int, default=64)

    args = parser.parse_args()
    dir_dataset = args.dir_dataset
    dir_model = args.dir_model
    number_epochs = int(args.number_epochs)
    batch_size = args.batch_size
    n_filters = args.n_filters

    # load the dataset
    motor_data, sensor_data, number_samples, height, width, number_channels, n_joints = load_data(dir_dataset=dir_dataset)

    # create the network
    network = SensoriMotorPredictionNetwork(n_joints, height, width, n_filter=n_filters)

    # train the network
    network.train(motor_data, sensor_data, dir_model=dir_model, n_epochs=number_epochs, batch_size=batch_size)

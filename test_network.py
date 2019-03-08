#!/usr/bin/env python
# coding: utf-8

import os
import glob
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import json
from sklearn import mixture
from scipy import interpolate
from tools import Normalizer, create_checkerboard, load_data
import matplotlib.pyplot as plt
plt.ion()


# todo
def generate_video():
    pass
    # import tensorflow as tf
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # import numpy as np
    # from scipy import interpolate
    # import cv2
    # import glob
    # import os
    #
    # ## PARAMETERS ##
    # network_dest = ".saved_models/pred_error/qibullet_arm/Abd_RGB_CLEAN_INPAINTING"
    # dataset_dest = ".datasets/qibullet_arm/Abd_RGB_CLEAN_INPAINTING"
    #
    # # reload the network
    # sess = tf.Session()
    # graph = tf.get_default_graph()
    # with graph.as_default():
    #     with sess.as_default():
    #         # restore the model
    #         saver = tf.train.import_meta_graph(network_dest + "/network.ckpt.meta")
    #         saver.restore(sess, tf.train.latest_checkpoint(network_dest + "/"))
    #         # for op in graph.get_operations():
    #         #     print(op.name)
    #         # doing prediction
    #         input = graph.get_tensor_by_name("x:0")
    #         output_mean = graph.get_tensor_by_name("image2/Relu:0")
    #         output_std = graph.get_tensor_by_name("mask/Relu:0")
    #         #
    #         # create a trajectory in the motor space
    #         anchors = 2 * np.random.rand(50, 4) - 1
    #         trajectory = np.full((2000, 4), np.nan)
    #         for k in range(4):
    #             tck = interpolate.splrep(np.linspace(0, 1, anchors.shape[0]), anchors[:, k])
    #             trajectory[:, k] = interpolate.splev(np.linspace(0, 1, trajectory.shape[0]), tck)
    #         #
    #         # prepare the video writer
    #         folder_video = ".temp/video"
    #         video = cv2.VideoWriter(filename=folder_video + "/video.avi", fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=24, frameSize=(800, 600))
    #         #
    #         fig = plt.figure(1, figsize=(8, 6))
    #         ax0 = fig.add_subplot(231, projection="3d")
    #         ax1 = fig.add_subplot(234, projection="3d")
    #         ax2 = fig.add_subplot(232)
    #         ax3 = fig.add_subplot(233)
    #         ax4 = fig.add_subplot(235)
    #         ax5 = fig.add_subplot(236)
    #         #
    #         for k in range(trajectory.shape[0]):
    #             #
    #             position = trajectory[[k], :]
    #             curr_image2, curr_error = sess.run([output_mean, output_std], feed_dict={input: position})
    #
    #             if k == 0:
    #                 # create checkerboard background
    #                 height, width = curr_image2.shape[1], curr_image2.shape[2]
    #                 square_light_gray = np.full((5, 5, 3), 250)
    #                 square_dark_gray = np.full((5, 5, 3), 200)
    #                 checker_square = np.vstack((np.hstack((square_dark_gray, square_light_gray)), np.hstack((square_light_gray, square_dark_gray))))
    #                 checkerboard = np.tile(checker_square,
    #                                        (int(np.ceil(height / checker_square.shape[0])), int(np.ceil(width / checker_square.shape[1])), 1))
    #                 checkerboard = checkerboard[:height, :width, :]
    #
    #             ax0.cla()
    #             ax0.set_title('$m_1, m_2, m_3$')
    #             ax0.plot(trajectory[max(0, k - 48):k, 0], trajectory[max(0, k - 48):k, 1], trajectory[max(0, k - 48):k, 2], 'b-')
    #             ax0.plot(trajectory[k - 1:k, 0], trajectory[k - 1:k, 1], trajectory[k - 1:k, 2], 'ro')
    #             ax0.set_xlim(-1, 1)
    #             ax0.set_ylim(-1, 1)
    #             ax0.set_zlim(-1, 1)
    #             ax0.set_xticklabels([])
    #             ax0.set_yticklabels([])
    #             ax0.set_zticklabels([])
    #
    #             ax1.cla()
    #             ax1.set_title('$m_2, m_3, m_4$')
    #             ax1.plot(trajectory[max(0, k - 48):k, 1], trajectory[max(0, k - 48):k, 2], trajectory[max(0, k - 48):k, 3], 'b-')
    #             ax1.plot(trajectory[k - 1:k, 1], trajectory[k - 1:k, 2], trajectory[k - 1:k, 3], 'ro')
    #             ax1.set_xlim(-1, 1)
    #             ax1.set_ylim(-1, 1)
    #             ax1.set_zlim(-1, 1)
    #             ax1.set_xticklabels([])
    #             ax1.set_yticklabels([])
    #             ax1.set_zticklabels([])
    #
    #             ax2.cla()
    #             ax2.set_title("predicted image")
    #             ax2.imshow(curr_image2[0])
    #             ax2.axis("off")
    #
    #             ax3.cla()
    #             ax3.set_title("predicted error")
    #             ax3.imshow(curr_error[0], cmap='gray')
    #             ax3.axis("off")
    #
    #             mask = curr_error[0] < 0.056
    #             ax4.cla()
    #             ax4.set_title("mask")
    #             ax4.imshow(mask.astype(float), cmap='gray')
    #             ax4.axis("off")
    #
    #             masked_image = np.dstack((curr_image2[0], np.mean(mask, axis=2)))
    #             ax5.cla()
    #             ax5.set_title("masked prediction")
    #             ax5.imshow(checkerboard)
    #             ax5.imshow(masked_image)
    #             ax5.axis("off")
    #             #
    #             plt.show(block=False)
    #             fig.savefig(folder_video + "/img.png")
    #             plt.pause(0.001)
    #
    #             # write frame
    #             image = cv2.imread(folder_video + "/img.png")
    #             video.write(image)
    #
    # cv2.destroyAllWindows()
    # video.release()
    #
    # for file in sorted(glob.glob(folder_video + "/img*.png")):
    #     os.remove(file)


def reconstruct_training_data(dir_model="model/trained", dir_dataset="dataset/generated/combined", indexes=6):

    # check directories
    if not os.path.exists(dir_model):
        print("Error: incorrect path to the model")
        return
    if not os.path.exists(dir_dataset):
        print("Error: incorrect path to the dataset")
        return

    # check the dataset
    files_list = sorted(glob.glob(dir_dataset + "/*.png"))
    n_samples = len(files_list)
    if n_samples == 0:
        print("Error: the directory doesn't contain any png image.")
        return

    # # draw training samples to test
    # if type(indexes) == int:
    #     indexes = np.random.choice(n_samples, indexes)
    #
    # # load, normalize, and subsample the motor data
    # with open(dir_dataset + "/positions.txt", "r") as file:
    #     m = np.array(json.load(file))
    # m_normalizer = Normalizer(low=-1, high=1)
    # m = m_normalizer.fit_transform(m)
    # m = m[indexes, :]
    #
    # # load and normalize the necessary sensory data
    # image = plt.imread(files_list[0])
    # height, width = image.shape[0], image.shape[1]
    # s = np.full((len(indexes), height, width, 3), np.nan)
    # for i, ind in enumerate(indexes):
    #     file = files_list[ind]
    #     s[i, :, :, :] = plt.imread(file)
    # s_normalizer = Normalizer(low=0, high=1, min_data=0, max_data=1)  # identity mapping in this case, as the pixel values are already in [0, 1]
    # s = s_normalizer.transform(s)

    # load the data
    m, s = load_data(dir_dataset)
    number_samples, height, width, number_channels = s.shape
    number_joints = m.shape[1]

    # draw training samples to test
    if type(indexes) == int:
        indexes = np.random.choice(n_samples, indexes)

    m_normalizer = Normalizer(low=-1, high=1)
    m = m_normalizer.fit_transform(m)
    m = m[indexes, :]

    s_normalizer = Normalizer(low=0, high=1, min_data=0, max_data=1)  # identity mapping in this case, as the pixel values are already in [0, 1]
    s = s_normalizer.transform(s)
    s = s[indexes, :]


    # reload the network
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(dir_model + "/network.ckpt.meta")
    graph = tf.get_default_graph()
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint(dir_model + "/"))

    # recover the input and outputs
    # input_motor = graph.get_tensor_by_name("joint_config:0")
    # output_image = graph.get_tensor_by_name("image/Relu:0")
    # output_error = graph.get_tensor_by_name("mask/Relu:0")
    input_motor = graph.get_tensor_by_name("x:0")
    output_image = graph.get_tensor_by_name("image2/Relu:0")
    output_error = graph.get_tensor_by_name("mask/Relu:0")

    # create the checkerboard
    checkerboard = create_checkerboard(height, width)

    for i, ind in enumerate(indexes):

        # ground truth image
        gt_image = s[i, :, :, :]

        # predict image
        curr_image = sess.run(output_image, feed_dict={input_motor: m[[i], :]})
        curr_image = curr_image[0]
        curr_image = s_normalizer.reconstruct(curr_image)  # identity mapping in this case, as the pixel values are already in [0, 1]

        # predict error
        curr_error = sess.run(output_error, feed_dict={input_motor: m[[i], :]})
        curr_error = curr_error[0]

        # build mask
        curr_mask = (curr_error > 0.056).astype(float)

        # build the masked image
        curr_masked_image = np.dstack((curr_image, 1 - np.mean(curr_mask, axis=2)))

        # display
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(233)
        ax4 = fig.add_subplot(235)
        ax5 = fig.add_subplot(236)
        #
        fig.suptitle('sample {}'.format(ind), fontsize=12)
        #
        ax1.set_title("ground-truth image")
        ax1.imshow(gt_image)
        ax1.axis("off")
        #
        ax2.set_title("predicted image")
        ax2.imshow(curr_image)
        ax2.axis("off")
        #
        ax3.set_title("predicted error")
        ax3.imshow(curr_error)
        ax3.axis("off")
        #
        ax4.set_title("mask")
        ax4.imshow(curr_mask)
        ax4.axis("off")
        #
        ax5.set_title('masked predicted image')
        ax5.imshow(checkerboard)
        ax5.imshow(curr_masked_image)
        ax5.axis("off")

    sess.close()
    plt.show(block=False)
    plt.pause(0.001)


def evaluate_mask(dir_model="model/trained", dir_green_dataset="dataset/generated/green", indexes=6):

    # check directories
    if not os.path.exists(dir_model):
        print("Error: incorrect path to the model")
        return
    if not os.path.exists(dir_green_dataset):
        print("Error: incorrect path to the dataset of images with green background")
        return

    # check the dataset
    files_list = sorted(glob.glob(dir_green_dataset + "/*.png"))
    n_samples = len(files_list)
    if n_samples == 0:
        print("Error: the directory doesn't contain any png image.")
        return

    # # draw training samples to test
    # if type(indexes) == int:
    #     indexes = np.random.choice(n_samples, indexes)
    #
    # # load, normalize, and subsample the motor data
    # with open(dir_green_dataset + "/positions.txt", "r") as file:
    #     m = np.array(json.load(file))
    # m_normalizer = Normalizer(low=-1, high=1)
    # m = m_normalizer.fit_transform(m)
    #
    # # load and normalize the necessary sensory data
    # image = plt.imread(files_list[0])
    # height, width = image.shape[0], image.shape[1]
    # s = np.full((n_samples, height, width, 3), np.nan)
    # for i in range(n_samples):
    #     file = files_list[i]
    #     s[i, :, :, :] = plt.imread(file)
    # s_normalizer = Normalizer(low=0, high=1, min_data=0, max_data=1)  # identity mapping in this case, as the pixel values are already in [0, 1]
    # s = s_normalizer.transform(s)

    # load the data
    m, s = load_data(dir_dataset)
    number_samples, height, width, number_channels = s.shape
    number_joints = m.shape[1]

    # draw training samples to test
    if type(indexes) == int:
        indexes = np.random.choice(n_samples, indexes)

    m_normalizer = Normalizer(low=-1, high=1)
    m = m_normalizer.fit_transform(m)
    m = m[indexes, :]

    s_normalizer = Normalizer(low=0, high=1, min_data=0, max_data=1)  # identity mapping in this case, as the pixel values are already in [0, 1]
    s = s_normalizer.transform(s)
    s = s[indexes, :]

    # reload the network
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(dir_model + "/network.ckpt.meta")
    graph = tf.get_default_graph()
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint(dir_model + "/"))

    # recover the input and outputs
    # input_motor = graph.get_tensor_by_name("joint_config:0")
    # output_image = graph.get_tensor_by_name("image/Relu:0")
    # output_error = graph.get_tensor_by_name("mask/Relu:0")
    input_motor = graph.get_tensor_by_name("x:0")
    output_image = graph.get_tensor_by_name("image2/Relu:0")
    output_error = graph.get_tensor_by_name("mask/Relu:0")

    # create a checkerboard
    checkerboard = create_checkerboard(height, width)

    # track all matches over the training set
    all_mask_match = []
    all_appearance_match = []

    for ind in range(n_samples):

        # image with green background
        green_image = s[ind, :, :, :]

        # mask of the green background
        where_green = ((green_image[:, :, 0] == 0) & (abs(green_image[:, :, 1] - 141/255) <= 1e-3) & (green_image[:, :, 2] == 0)).astype(float)
        where_green = np.repeat(where_green[:, :, np.newaxis], 3, axis=2)  # copy on all three RGB channels

        # predict image
        curr_image = sess.run(output_image, feed_dict={input_motor: m[[ind], :]})
        curr_image = curr_image[0]
        curr_image = s_normalizer.reconstruct(curr_image)  # identity mapping in this case, as the pixel values are already in [0, 1]

        # predict error
        curr_error = sess.run(output_error, feed_dict={input_motor: m[[ind], :]})
        curr_error = curr_error[0]

        # build mask
        curr_mask = (curr_error > 0.056).astype(float)

        # build the masked predicted image
        curr_masked_image = np.dstack((curr_image, 1 - np.mean(curr_mask, axis=2)))

        # build the masked green-backgroundimage
        curr_masked_green_image = np.dstack((green_image, 1 - np.mean(curr_mask, axis=2)))

        # error between green background mask and predicted mask
        error_mask_green = where_green - curr_mask

        # matching between the two masks
        mask_match = 1 - np.sum(np.abs(error_mask_green)) / np.prod(error_mask_green.shape)

        # error between the arm appearance under the predicted mask
        image_error = np.abs(green_image - curr_image)
        image_error = image_error * (1 - curr_mask)

        # matching between the appearances between the predicted mask
        if not np.sum(1 - curr_mask) == 0:
            appearance_match = 1 - np.sum(np.abs(image_error)) / np.sum(1 - curr_mask)
        else:
            appearance_match = 1

        # store the matches
        all_mask_match.append(mask_match)
        all_appearance_match.append(appearance_match)

        if ind in indexes:
            # display
            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot(231)
            ax2 = fig.add_subplot(232)
            ax3 = fig.add_subplot(233)
            ax4 = fig.add_subplot(234)
            ax5 = fig.add_subplot(235)
            ax6 = fig.add_subplot(236)
            #
            fig.suptitle('sample {}'.format(ind), fontsize=12)
            #
            ax1.set_title("ground-truth background")
            ax1.imshow(green_image * where_green)
            ax1.axis("off")
            #
            ax2.set_title("predicted mask")
            ax2.imshow(curr_mask)
            ax2.axis("off")
            #
            ax3.set_title("mask error: {:.2f}%".format(100 * mask_match), fontsize=11)
            ax3.imshow(error_mask_green / 2 + 0.5)
            ax3.axis("off")
            #
            #
            ax4.set_title("masked ground-truth")
            ax4.imshow(checkerboard)
            ax4.imshow(curr_masked_green_image)
            ax4.axis("off")
            #
            ax5.set_title("masked prediction")
            ax5.imshow(checkerboard)
            ax5.imshow(curr_masked_image)
            ax5.axis("off")
            #
            ax6.set_title("appearance error: {:2f}%".format(100 * appearance_match), fontsize=11)
            ax6.imshow(checkerboard)
            ax6.imshow(curr_masked_image)
            ax6.axis("off")

    sess.close()

    # print the states
    print("mask match = {mean} +/- {std}".format(mean=np.mean(all_mask_match), std=np.std(all_mask_match)))
    print("appearance match = {mean} +/- {std}".format(mean=np.mean(all_appearance_match), std=np.std(all_appearance_match)))

    plt.show(block=False)
    plt.pause(0.001)


def fit_gmm(dir_dataset="dataset/generated.combined", dir_model="model/trained", indexes=100):

    # check directories
    if not os.path.exists(dir_model):
        print("Error: incorrect path to the model")
        return
    if not os.path.exists(dir_dataset):
        print("Error: incorrect path to the dataset of images with green background")
        return

    # check the dataset
    files_list = sorted(glob.glob(dir_dataset + "/*.png"))
    n_samples = len(files_list)
    if n_samples == 0:
        print("Error: the directory doesn't contain any png image.")
        return

    # # draw training samples to test
    # if type(indexes) == int:
    #     indexes = np.random.choice(n_samples, indexes)
    #
    # # load, normalize, and subsample the motor data
    # with open(dir_dataset + "/positions.txt", "r") as file:
    #     m = np.array(json.load(file))
    # m_normalizer = Normalizer(low=-1, high=1)
    # m = m_normalizer.fit_transform(m)
    # m = m[indexes, :]

    # load the data
    m, _ = load_data(dir_dataset)
    number_joints = m.shape[1]

    # draw training samples to test
    if type(indexes) == int:
        indexes = np.random.choice(n_samples, indexes)

    m_normalizer = Normalizer(low=-1, high=1)
    m = m_normalizer.fit_transform(m)
    m = m[indexes, :]


    # reload the network
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(dir_model + "/network.ckpt.meta")
    graph = tf.get_default_graph()
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint(dir_model + "/"))

    # recover the input and outputs
    # input_motor = graph.get_tensor_by_name("joint_config:0")
    # output_error = graph.get_tensor_by_name("mask/Relu:0")
    input_motor = graph.get_tensor_by_name("x:0")
    output_error = graph.get_tensor_by_name("mask/Relu:0")

    # initialize list
    all_pred_errors = []

    for i, ind in enumerate(indexes):

        # predict error
        curr_error = sess.run(output_error, feed_dict={input_motor: m[[i], :]})
        curr_error = curr_error[0]

        # append errors
        all_pred_errors = all_pred_errors + list(curr_error.flatten())

    # fit a 2-GMM model
    all_pred_errors =  np.array(all_pred_errors).reshape(-1, 1)
    gmm_model = mixture.GaussianMixture(n_components=2, n_init=5)
    gmm_model.fit(all_pred_errors)

    # find the intersection of the two gaussians
    x = np.linspace(-0.05, 0.3, 1000).reshape(-1, 1)
    lp = gmm_model.score_samples(x)  # log probability
    p = gmm_model.predict_proba(x)  # class prediction
    diff = np.abs(p[:, 0] - p[:, 1])
    cross_index = np.argmin(diff)
    threshold = x[cross_index, 0]

    print("Estimated error threshold: {:.3f}".format(threshold))

    # display the histogram and optimizes gaussians
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #
    ax.hist(all_pred_errors[:, 0], bins=100, normed=True, color="blue", rwidth=0.8, label="errors")
    ax.plot(x, np.exp(lp), 'r-', label="GMM")
    ax.legend(loc="upper left")
    #
    ax2 = ax.twinx()
    ax2.plot(x, p[:, 0], 'c--', label="Proba comp 1")
    ax2.plot(x, p[:, 1], 'g--', label="Proba comp 2")
    ax2.set_ylim([0, 1.2])
    ax2.legend(loc="upper right")
    #
    plt.show(block=False)
    plt.pause(0.001)

    return threshold


# todo
def explore_joint_space():
    pass
    # ###########################
    # # CHECK ON REGULAR SAMPLING
    # ###########################
    # #
    # # reload the network
    # sess = tf.Session()
    # graph = tf.get_default_graph()
    # with graph.as_default():
    #     with sess.as_default():
    #         # restore the model
    #         saver = tf.train.import_meta_graph(network_dest + "/network.ckpt.meta")
    #         saver.restore(sess, tf.train.latest_checkpoint(network_dest + "/"))
    #         # for op in graph.get_operations():
    #         #     print(op.name)
    #         # doing prediction
    #         input = graph.get_tensor_by_name("x:0")
    #         output_mean = graph.get_tensor_by_name("image2/Relu:0")
    #         output_std = graph.get_tensor_by_name("mask/Relu:0")
    #         #
    #         fig = plt.figure(1, figsize=(10, 3))
    #         #
    #         position_ref = np.zeros((1, 4))
    #         for index, val in enumerate(np.linspace(-1, 1, 6)):
    #             ax1 = fig.add_axes([index / 6, 1 / 2, 1 / 6 - 0.01, 1 / 2 - 0.01])
    #             ax2 = fig.add_axes([index / 6, 0 / 2, 1 / 6 - 0.01, 1 / 2 - 0.01])
    #             #
    #             position = position_ref + [0, 0, 0, val]
    #             curr_image2, curr_error = sess.run([output_mean, output_std], feed_dict={input: position})
    #             #
    #             # ax1.set_title('real output')
    #             ax1.imshow(curr_image2[0])
    #             ax1.axis("off")
    #             # ax2.set_title('image2')
    #             ax2.imshow(curr_error[0])
    #             ax2.axis("off")
    #             plt.show(block=False)
    #         #
    #         fig.savefig(".temp/testing2.svg")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-dm", "--dir_model", dest="dir_model", help="path to the model", default="model/trained")
    parser.add_argument("-dd", "--dir_dataset", dest="dir_dataset", help="path to training dataset", default="dataset/generated/combined")

    args = parser.parse_args()
    dir_model = args.dir_model
    dir_dataset = args.dir_dataset

####
    dir_model = "../_old_learn-masked-body-image/.saved_models/pred_error/qibullet_arm/Abd_RGB_CLEAN_INPAINTING"
    dir_dataset = ".dataset/generated/combined"
    dir_green_dataset = ".dataset/generated/combined"
####

    reconstruct_training_data(dir_model=dir_model, dir_dataset=dir_dataset, indexes=3)
    evaluate_mask(dir_model=dir_model, dir_green_dataset=dir_green_dataset, indexes=3)
    fit_gmm(dir_dataset=dir_dataset, dir_model=dir_model, indexes=100)

    plt.show(block=True)

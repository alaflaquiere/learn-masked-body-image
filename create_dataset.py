#!/usr/bin/env python
# coding: utf-8

import os
import sys
from argparse import ArgumentParser
import cv2
import pybullet as pb
import pybullet_data
from qibullet import PepperVirtual
from qibullet.camera import Camera
from qibullet.base_controller import BaseController
import numpy as np
import json
import glob


# TODO: DO I USE A CUSTOM PEPPERVIRTUAL??
# TODO: define the joints to be explored as argument
# todo: add the size of the output image as argument
# todo: send a warnign when the image_size is bigger than the camera resolution
# todo: select saving images in binary of as individual images

def create_dataset(number_images=8000, image_size=(60, 80), dir_dataset="dataset", dir_bkgd="dataset/background_dataset", keep_green=False):

    # check directories
    if not os.path.exists(dir_bkgd):
        print("Error: incorrect path for the background dataset")
        sys.exit(0)

    if os.path.exists(dir_dataset):
        ans = input(" ".join(["> The folder", dir_dataset, "already exists; do you want to overwrite its content? [y,n]: "]))
        if ans is not "y":
            print("exiting the program")
            sys.exit(0)

    dir_combined = dir_dataset + "/combined"
    if not os.path.exists(dir_combined):
        os.makedirs(dir_combined)
    if keep_green:
        dir_green = dir_dataset + "/green"
        if not os.path.exists(dir_green):
            os.makedirs(dir_green)

    # list the background images
    bkgd_images_list = glob.glob(dir_bkgd + "/*.png")

    # initialize the simulator
    pb.connect(pb.GUI)

    # configure OpenGL settings
    pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
    pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

    # command the server to run in realtime
    pb.setRealTimeSimulation(1)

    # set gravity
    pb.setGravity(0, 0, -9.81)

    # use pybullet own data files
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.loadMJCF("mjcf/ground_plane.xml")

    # create the Pepper robot
    pepper = PepperVirtual()
    pepper.loadRobot([0, 0, 0], [0, 0, 0, 1])
    pepper_ID = 1

    # define the initial posture
    joint_parameters = list()
    for name, joint in pepper.joint_dict.items():
        if "Finger" not in name and "Thumb" not in name:
            if name == "HeadPitch":
                pepper.setAngles(name, -0.253, 1.0)
            elif name == "HeadYaw":
                pepper.setAngles(name, -0.7, 1.0)
            else:
                pepper.setAngles(name, 0, 1.0)
            joint_parameters.append((len(joint_parameters), name))

    # subscribe to the camera
    pepper.subscribeCamera(PepperVirtual.ID_CAMERA_BOTTOM)
    camera_resolution = (240, 320)

    # turn the Pepper around to avoid shadows
    pb.resetBasePositionAndOrientation(pepper_ID, posObj=[0, 0, 0], ornObj=pb.getQuaternionFromEuler((0, 0, 1.25 * np.pi)))

    # add a green wall in front of Pepper
    wall = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.1, 3, 1], rgbaColor=[0, 1, 0, 1])
    pb.createMultiBody(baseVisualShapeIndex=wall, basePosition=[-0.65, 0, 1])

    # allocate memory
    joints_configs = np.full((number_images, 4), np.nan)
    images_green = np.full((number_images, int(image_size[0]), int(image_size[1]), 3), np.nan)
    images_composite = np.full((number_images, int(image_size[0]), int(image_size[1]), 3), np.nan)

    # generate the images
    for t in range(number_images):

        if t % max(1, int(number_images/100)) == 0:
            print("\r{:3.0f}%".format((t+1)/number_images * 100), end="")

        # draw random joints configurations
        RShoulRoll_val = float(np.random.rand(1) - 1)
        RElbRoll_val = float(np.random.rand(1))
        RElbYaw_val = float(2 * np.random.rand(1) - 1)
        RShoulPitch_val = float(2 * np.random.rand(1) - 1)

        # set the posture
        pepper.setAngles("RShoulderRoll", RShoulRoll_val, 1)
        pepper.setAngles("RElbowRoll", RElbRoll_val, 1)
        pepper.setAngles("RElbowYaw", RElbYaw_val, 1)
        pepper.setAngles("RShoulderPitch", RShoulPitch_val, 1)

        # wait for the movement to be finished
        cv2.waitKey(800)

        # get the camera input and display it
        img = pepper.getCameraFrame()
        cv2.imshow("bottom camera", img)
        cv2.waitKey(1)

        # save the joint_configuration
        joints_configs[t, :] = [RShoulRoll_val, RElbRoll_val, RElbYaw_val, RShoulPitch_val]

        # scale the image down
        image_green = cv2.resize(img, dsize=image_size[::-1], interpolation=cv2.INTER_NEAREST)

        # save the green image
        images_green[t, :, :, :] = image_green

        # draw a random background
        background = cv2.imread(np.random.choice(bkgd_images_list))

        # rescale the background if necessary
        if not background.shape == image_size:
            background = cv2.resize(background, dsize=image_size[::-1], interpolation=cv2.INTER_NEAREST)

        # fill the green background with the background image
        to_fill = (image_green[:, :, 0] == 0) & (image_green[:, :, 1] == 141) & (image_green[:, :, 2] == 0)
        image_green[to_fill, :] = background[to_fill, :]

        # save the composite image
        images_composite[t, :, :, :] = image_green

        # display the composite image
        cv2.imshow('mit', image_green)
        cv2.waitKey(1)

    # stop the simulation
    for camera in Camera._getInstances():
        camera._resetActiveCamera()
    for controller in BaseController._getInstances():
        controller._terminateController()
    pb.disconnect()

    # save the data
    print("saving the data...")

    with open(dir_combined + "/positions.json", 'w') as file:
        json.dump(joints_configs.tolist(), file)
    with open(dir_combined + "/images.json", 'w') as file:
        json.dump(images_composite.tolist(), file)

    if keep_green:
        with open(dir_green + "/positions.json", 'w') as file:
            json.dump(joints_configs.tolist(), file)
        with open(dir_green + "/images.json", 'w') as file:
            json.dump(images_green.tolist(), file)

    print("done")

    sys.exit(0)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-n", "--n_samples", dest="number_images", help="number of images to generate", type=int, default=8000)
    parser.add_argument("-s", "--size", dest="image_size", help="image's height and width", nargs=2, type=int, default=[60, 80])
    parser.add_argument("-dd", "--dir_dataset", dest="dir_dataset", help="directory in which to save the data", default="dataset/combined")
    parser.add_argument("-db", "--dir_bkgd", dest="dir_bkgd", help="directory of background images", default="dataset/background_dataset")
    parser.add_argument("-g", "--green", dest="keep_green", help="flag to store the raw images with the green background", type=bool, default=False)

    # get arguments
    args = parser.parse_args()
    number_images = args.number_images
    image_size = tuple(args.image_size)
    dir_dataset = args.dir_dataset
    dir_bkgd = args.dir_bkgd
    keep_green = args.keep_green

    # run the simulation
    create_dataset(number_images=number_images, image_size=image_size, dir_dataset=dir_dataset, dir_bkgd=dir_bkgd, keep_green=keep_green)

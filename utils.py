from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
import os
import tensorflow as tf
import pathlib
import requests
import zipfile
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import random


class Dataset:
    # define function to load train, test, and validation datasets
    @staticmethod
    def load_dataset(path_, no_categories):
        data = load_files(path_)
        files = np.array(data["filenames"])
        labels = np_utils.to_categorical(np.array(data["target"]), no_categories)
        return files, labels


class Nature:
    pass


class Animal:
    @staticmethod
    def download_dataset(
        url="https://cvml.ista.ac.at/AwA2/AwA2-data.zip",
        download=False,
        directory="datasets",
    ):
        # Download and extract the AWA2 dataset
        if download:
            response = requests.get(url, stream=True)

            fname = "AWA2-data.zip"
            with open(fname, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

            # Extract the contents of the zip file to a temporary directory
            with zipfile.ZipFile(fname) as zip_file:
                zip_file.extractall(path="datasets")

            os.rename(
                os.path.join(directory, "Animals_with_Attributes2"),
                os.path.join(directory, "animals"),
            )

    @staticmethod
    def datasets(directory=os.path.join("datasets", "animals", "JPEGImages")):
        subdirs = [
            name
            for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))
        ]

        no_categories = len(set(subdirs))
        files_dataset, labels_dataset = Dataset.load_dataset(directory, no_categories)

        files_dict = dict.fromkeys(("train", "valid", "test"))
        labels_dict = dict.fromkeys(("train", "valid", "test"))

        (
            files_dict["train"],
            files_dict["valid"],
            labels_dict["train"],
            labels_dict["valid"],
        ) = train_test_split(
            files_dataset, labels_dataset, test_size=0.2, random_state=42
        )

        (
            files_dict["valid"],
            files_dict["test"],
            labels_dict["valid"],
            labels_dict["test"],
        ) = train_test_split(
            files_dict["valid"], labels_dict["valid"], test_size=0.5, random_state=42
        )

        print(f"There are {no_categories} total human categories.")
        print(
            f'There are {len(np.hstack([files_dict["train"], files_dict["valid"], files_dict["test"]]))} total human images.\n'
        )
        print(
            f'There are {len(files_dict["train"])} training files and {len(labels_dict["train"])} training labels.'
        )
        print(
            f'There are {len(files_dict["valid"])} validation files and {len(labels_dict["valid"])} validation labels.'
        )
        print(
            f'There are {len(files_dict["test"])} test files and {len(labels_dict["test"])} test labels.'
        )

        return files_dict, labels_dict


class Dog:
    @staticmethod
    def datasets(directory=os.path.join("datasets", "dogImages")):
        dataset_keys = ("train", "valid", "test")
        files_dict = dict.fromkeys(dataset_keys)
        labels_dict = dict.fromkeys(dataset_keys)

        subdirs = []
        for k in dataset_keys:
            subdirs.extend(
                [
                    name
                    for name in os.listdir(os.path.join(directory, k))
                    if os.path.isdir(os.path.join(os.path.join(directory, k), name))
                ]
            )

        no_categories = len(set(subdirs))
        # load train, test, and validation datasets
        files_dict["train"], labels_dict["train"] = Dataset.load_dataset(
            os.path.join(directory, "train"), no_categories
        )
        files_dict["valid"], labels_dict["valid"] = Dataset.load_dataset(
            os.path.join(directory, "valid"), no_categories
        )
        files_dict["test"], labels_dict["test"] = Dataset.load_dataset(
            os.path.join(directory, "test"), no_categories
        )

        # load list of dog names
        dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

        # print statistics about the dataset
        print(f"There are {no_categories} total dog categories.")
        print(
            f'There are {len(np.hstack([files_dict["train"], files_dict["valid"], files_dict["test"]]))} total dog images.\n'
        )
        print(
            f'There are {len(files_dict["train"])} training files and {len(labels_dict["train"])} training labels.'
        )
        print(
            f'There are {len(files_dict["valid"])} validation files and {len(labels_dict["valid"])} validation labels.'
        )
        print(
            f'There are {len(files_dict["test"])} test files and {len(labels_dict["test"])} test labels.'
        )

        return files_dict, labels_dict, dog_names


class Human:
    @staticmethod
    def download_dataset(
        url="http://vis-www.cs.umass.edu/lfw/lfw.tgz", directory="datasets"
    ):
        # Download and extract the LFW dataset
        data_dir = tf.keras.utils.get_file(
            "lfw", url, untar=True, cache_dir=directory, cache_subdir="lfw"
        )
        print(data_dir)
        print(pathlib.Path(data_dir))

    @staticmethod
    def datasets(directory=os.path.join("datasets", "lfw")):
        subdirs = [
            name
            for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))
        ]

        no_categories = len(set(subdirs))
        files_dataset, labels_dataset = Dataset.load_dataset(directory, no_categories)

        files_dict = dict.fromkeys(("train", "valid", "test"))
        labels_dict = dict.fromkeys(("train", "valid", "test"))

        (
            files_dict["train"],
            files_dict["valid"],
            labels_dict["train"],
            labels_dict["valid"],
        ) = train_test_split(
            files_dataset, labels_dataset, test_size=0.2, random_state=42
        )

        (
            files_dict["valid"],
            files_dict["test"],
            labels_dict["valid"],
            labels_dict["test"],
        ) = train_test_split(
            files_dict["valid"], labels_dict["valid"], test_size=0.5, random_state=42
        )

        print(f"There are {no_categories} total human categories.")
        print(
            f'There are {len(np.hstack([files_dict["train"], files_dict["valid"], files_dict["test"]]))} total human images.\n'
        )
        print(
            f'There are {len(files_dict["train"])} training files and {len(labels_dict["train"])} training labels.'
        )
        print(
            f'There are {len(files_dict["valid"])} validation files and {len(labels_dict["valid"])} validation labels.'
        )
        print(
            f'There are {len(files_dict["test"])} test files and {len(labels_dict["test"])} test labels.'
        )

        return files_dict, labels_dict

    @staticmethod
    def no_faces_detector(img_path):
        # extract pre-trained face detector
        face_cascade = cv2.CascadeClassifier(
            "haarcascades/haarcascade_frontalface_alt.xml"
        )
        img = cv2.imread(img_path)
        print(f"Shape of image: {img.shape}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return face_cascade.detectMultiScale(gray)

    @staticmethod
    def face_detector(img_path):
        faces = Human.no_faces_detector(img_path)
        return len(faces) > 0

    @staticmethod
    def write_resized_img(img, x_pixels, y_pixels, out_path):
        if img.shape[0] < x_pixels and img.shape[1] < y_pixels:
            interpol = cv2.INTER_CUBIC
        elif img.shape[0] > x_pixels and img.shape[1] > y_pixels:
            interpol = cv2.INTER_AREA
        else:
            interpol = cv2.INTER_LINEAR
        resized_img = cv2.resize(img, (x_pixels, y_pixels), interpolation=interpol)
        cv2.imwrite(out_path, resized_img)

    @staticmethod
    def store_raw_images(
        neg_samples, # choose samples from Animal (and maybe Nature dataset), make sure it is training data
        pos_samples, # choose samples from LFW dataset, make sure it is training data
        x_pixels=250,
        y_pixels=250,
        neg_out_dir=os.path.join("datasets", "neg"),
        pos_out_dir=os.path.join("datasets", "pos"),
    ):
        # NEGATIVE SAMPLES
        if not os.path.exists(neg_out_dir):
            os.makedirs(neg_out_dir)

        pic_num = 1
        for img in neg_samples:
            neg_out_path = os.path.join(neg_out_dir, f"{pic_num}.jpg")
            Human.write_resized_img(img, x_pixels, y_pixels, neg_out_path)
            pic_num += 1

        # POSITIVE SAMPLES
        if not os.path.exists(pos_out_dir):
            os.makedirs(pos_out_dir)

        pic_num = 1
        for img in pos_samples:
            pos_out_path = os.path.join(pos_out_dir, f"{pic_num}.jpg")
            Human.write_resized_img(img, x_pixels, y_pixels, pos_out_path)
            pic_num += 1

    @staticmethod
    def test_faces_detector(img_path):
        img = cv2.imread(img_path)
        faces = Human.no_faces_detector(img_path)

        # get bounding box for each detected face
        for x, y, w, h in faces:
            # add bounding box to color image
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # convert BGR image to RGB for plotting
        cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # display the image, along with bounding box
        plt.imshow(cv_rgb)
        plt.show()

    @staticmethod
    def test2_faces_detector(human_files, dog_files, no_samples=1000):
        human_files_short = human_files[0:no_samples]
        dog_files_short = dog_files[0:no_samples]

        h_in_h_count = 0
        h_in_d_count = 0

        for human_files_1 in human_files_short:
            if Human.face_detector(human_files_1):
                h_in_h_count += 1
            else:
                print(
                    f"no human detected at index i = {np.where(human_files_1==human_files_short)[0][0]}"
                )

        for dog_files in dog_files_short:
            if Human.face_detector(dog_files):
                h_in_d_count += 1

        h_in_h_relative = h_in_h_count / (len(human_files_short))
        h_in_d_relative = h_in_d_count / (len(dog_files_short))
        print(f"percentage of detected humans in human images = {h_in_h_relative:.1%}")
        print(f"percentage of detected humans in dog images = {h_in_d_relative:.1%}")

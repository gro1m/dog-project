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
from tensorflow.keras.datasets import cifar100
from sklearn.model_selection import StratifiedShuffleSplit

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras.layers import Activation
from keras.models import Sequential

import itertools


class Dataset:
    # define function to load train, test, and validation datasets
    @staticmethod
    def load_dataset(path_, no_categories):
        data = load_files(path_)
        files = np.array(data["filenames"])
        labels = np_utils.to_categorical(np.array(data["target"]), no_categories)
        return files, labels


class Nature:
    @staticmethod
    def whole_dataset():
        files_dict, labels_dict = Nature.datasets()
        files_dataset = files_dict["train"].tolist()+files_dict["test"].tolist()+files_dict["valid"].tolist()
        labels_dataset = labels_dict["train"].tolist()+labels_dict["test"].tolist()+labels_dict["valid"].tolist()
        return files_dataset, labels_dataset

    @staticmethod
    def datasets():
        files_dict = dict.fromkeys(("train", "valid", "test"))
        labels_dict = dict.fromkeys(("train", "valid", "test"))

        # Load the CIFAR-100 data
        (files_dict["train"], labels_dict["train"]), (
            files_dict["valid"],
            labels_dict["valid"],
        ) = cifar100.load_data(label_mode="fine")


        files_dict["valid"], files_dict["test"],labels_dict["valid"], labels_dict["test"] = train_test_split(
            files_dict["valid"], labels_dict["valid"], test_size=0.5, random_state=42
        )

        return files_dict, labels_dict


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
    def whole_dataset(directory=os.path.join("datasets", "animals", "JPEGImages")):
        subdirs = [
            name
            for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))
        ]

        no_categories = len(set(subdirs))
        files_dataset, labels_dataset = Dataset.load_dataset(directory, no_categories)
        return files_dataset, labels_dataset, no_categories
    
    @staticmethod
    def datasets():
        files_dataset, labels_dataset, no_categories = Animal.whole_dataset()
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

        print(f"There are {no_categories} total animal ategories.")
        print(
            f'There are {len(np.hstack([files_dict["train"], files_dict["valid"], files_dict["test"]]))} total animal images.\n'
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
    def whole_dataset():
        files_dict, labels_dict, _ = Dog.datasets()
        files_dataset = files_dict["train"].tolist()+files_dict["test"].tolist()+files_dict["valid"].tolist()
        labels_dataset = labels_dict["train"].tolist()+labels_dict["test"].tolist()+labels_dict["valid"].tolist()
        return files_dataset, labels_dataset

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
    def whole_dataset(directory=os.path.join("datasets", "lfw")):
        subdirs = [
            name
            for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))
        ]

        no_categories = len(set(subdirs))
        files_dataset, labels_dataset = Dataset.load_dataset(directory, no_categories)
        return files_dataset, labels_dataset, no_categories

    @staticmethod
    def datasets():
        files_dataset, labels_dataset, no_categories = Human.whole_dataset()

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
    def no_faces_detector(img_path, haarcascade_filepath=os.path.join("datasets", "haarcascades", "haarcascade_frontalface_alt.xml")):
        # extract pre-trained face detector
        face_cascade = cv2.CascadeClassifier(haarcascade_filepath)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return face_cascade.detectMultiScale(gray), img.shape

    @staticmethod
    def face_detector(img_path):
        faces, img_shape = Human.no_faces_detector(img_path)
        return len(faces) > 0, img_shape

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
        neg_samples,  # choose samples from Animal (and maybe Nature dataset), make sure it is training data
        pos_samples,  # choose samples from LFW dataset, make sure it is training data
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
        faces, _ = Human.no_faces_detector(img_path)

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

        dict_h_in_h, dict_h_in_d = dict(), dict()

        for iter_count, human_files in enumerate(human_files_short):
            is_face, img_shape = Human.face_detector(human_files)
            if is_face:
                if img_shape in dict_h_in_h:
                    dict_h_in_h[img_shape] += 1
                else:
                    dict_h_in_h[img_shape] = 1
            else:
                print(
                    f"no human detected at index {iter_count} which corresponds to file = {human_files}"
                )

        for iter_count, dog_files in enumerate(dog_files_short):
            is_face, img_shape = Human.face_detector(dog_files)
            if is_face:
                print(
                    f"detected human in dog image at index {iter_count} which corresponds to file = {dog_files}"
                )
                if img_shape in dict_h_in_d:
                    dict_h_in_d[img_shape] += 1
                else:
                    dict_h_in_d[img_shape] = 1

        h_in_h_relative = sum(dict_h_in_h.values()) / (len(human_files_short))
        print(f"percentage of detected humans in human images = {h_in_h_relative:.1%}")

        h_in_d_relative = sum(dict_h_in_d.values())/ (len(dog_files_short))
        print(f"percentage of detected humans in dog images = {h_in_d_relative:.1%}")

class Face:
    @staticmethod
    def image_paths():
        list_human_files = [str(e) for e in Human.whole_dataset()[0]]
        list_array_nature = Nature.whole_dataset()[0]
        list_animal_files = [str(e) for e in Animal.whole_dataset()[0]]
        list_dog_files = [str(e) for e in Dog.whole_dataset()[0]]

        number_noface_images = len(list_array_nature+list_animal_files+list_dog_files)
        number_face_images = len(list_human_files)
        print(f"Number of 'no face' images = {number_noface_images}")
        print(f"Number of 'face' images = {number_face_images}")
        return list_array_nature, list_animal_files+list_dog_files+list_human_files, number_noface_images, number_face_images

    @staticmethod
    def labelled_data():
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        print("Read image paths...")
        list_array_nature, img_paths, number_noface_images, number_face_images = Face.image_paths()
        print("done.")
        print("Processing files with open cv...")
        data = list_array_nature+[cv2.imread(p) for p in img_paths]
        print("done.")
        print("Making labels...")
        labels = [np.concatenate((np.zeros(number_noface_images), np.ones(number_face_images)),axis=0)]
        print("done.")
        profiler.disable()
        profiler.print_stats(sort='cumulative')
        return data, labels

    @staticmethod
    def datasets(data=None, labels=None, test_size=0.1, val_size=0.1, normalization_factor=255):
        train_size = 1 - (test_size + val_size)
        test_val_size = 1 - train_size
        # split into 80% train data (first entry [0][0]) and 20% test data (second entry [0][1])
        stratSplit1 = StratifiedShuffleSplit(n_splits = 1, test_size   = test_val_size, train_size  = train_size,random_state=None)
        
        # split into total 10 % test data and total 10% validation data
        stratSplit2 = StratifiedShuffleSplit(n_splits=1,test_size = test_size/test_val_size,train_size = val_size/test_val_size,random_state=None)

        data,labels = Face.labelled_data()
        indices1 = stratSplit1.split(data, labels)

        train_indices, test_val_indices = indices1[0][0], indices1[0][1]
        X_train, labels_train  = [data[train_indices[idx1]] for idx1 in indices1], labels[train_indices]
        
        X_test_val, labels_test_val = [data[test_val_indices[idx1]] for idx1 in indices1], labels[test_val_indices]
        indices2 = list(stratSplit2.split(X_test_val,labels_test_val))
        test_indices, val_indices  = indices2[0][0], indices2[0][1]
        X_test, labels_test  = [data[test_indices[idx1]] for idx1 in indices1], labels[test_indices]
        X_val, labels_val  = [data[val_indices[idx1]] for idx1 in indices1], labels[val_indices]

        X_train, y_train = np.asarray(X_train).astype('float32')/normalization_factor, np.asarray(labels_train)
        X_test,  y_test  = np.asarray(X_test).astype('float32')/normalization_factor , np.asarray(labels_test)
        X_val, y_val = np.asarray(X_val).astype('float32')/normalization_factor, np.asarray(labels_val)

        print(f"Train      shapes ({X_train.shape}, {y_train.shape}):", )
        print(f"Validation shapes ({X_val.shape}, {y_val.shape}):")
        print(f"Test       shapes ({X_test.shape}, {y_test.shape})")

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def model():
        def conv2d(mdl, filters = 16, kernel_size = (3,3), strides = (1, 1),\
                padding = 'same', input_shape = None, name = 'conv'):
            if(input_shape == None):
                mdl.add(Conv2D(filters     = filters,\
                            kernel_size = kernel_size,\
                            strides     = strides,\
                            padding     = padding,\
                            name        = name))
            else:
                mdl.add(Conv2D(filters     = filters,\
                            kernel_size = kernel_size,\
                            strides     = strides,\
                            padding     = padding,\
                            input_shape = input_shape,\
                            name        = name))

        def maxpool2d(mdl, pool_size = (2,2), strides = (2,2), name = 'maxpool'):
            mdl.add(MaxPooling2D(pool_size, strides, name = name))

        def activation(mdl, activation = 'relu', name = 'activation'):
            mdl.add(Activation(activation, name = name))

        def gap2d(mdl,data_format = 'channels_last'):
            mdl.add(GlobalAveragingPooling2D(data_format))

        def flatten(mdl, name='flatten'):
            mdl.add(Flatten(name=name))

        def dense(mdl, filters, name='dense'):
            mdl.add(Dense(filters, name=name))

        def dropout(mdl, drop_prob = 0.2, name='dropout'):
            mdl.add(Dropout(drop_prob, name=name))
            
        def batchnormalize(mdl, name='batchnormalize'):
            mdl.add(BatchNormalization(name=name))

        faces_model = Sequential()

        ### TODO: Define your architecture.
        conv2d(faces_model, input_shape = (250,250,3), name='faces_conv2d_1')
        dropout(faces_model, name='faces_dropout_1')
        batchnormalize(faces_model,name='faces_batchnorm_1')
        activation(faces_model, name='faces_activation_1')
        maxpool2d(faces_model, name='faces_maxpool_1')
        conv2d(faces_model, filters = 32, name='faces_conv2d_2')
        dropout(faces_model, name='faces_dropout_2')
        batchnormalize(faces_model,name='faces_batchnorm_2')
        activation(faces_model, name='faces_activation_2')
        maxpool2d(faces_model, name='faces_maxpool_2')
        conv2d(faces_model, filters = 64, name='faces_conv2d_3')
        dropout(faces_model, name='faces_dropout_3')
        batchnormalize(faces_model,name='faces_batchnorm_3')
        activation(faces_model, name='faces_activation_3')
        maxpool2d(faces_model, name='faces_maxpool_3')
        flatten(faces_model, name='faces_flatten')
        dense(faces_model, filters = 1, name='faces_dense')
        batchnormalize(faces_model,name='faces_batchnorm_4')
        activation(faces_model, 'softmax', name='faces_activation_4')

        faces_model.summary()

        faces_model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
        return faces_model
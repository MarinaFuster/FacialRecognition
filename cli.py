import csv
import numpy as np
from numpy import genfromtxt

from data_loading import load_images
from pathlib import Path
from PIL import Image
from pyfiglet import Figlet
from image_resize import resizeImage
import os
import re


def init_cli():
    f = Figlet(font='slant')
    print("Welcome!")
    print(f.renderText('FaceRecognition'))
    print("In order to quit write exit.")
    print("IMPORTANT:")
    print("1. In order to make things simple, labels should be in the filename.")
    print("2. Images should be 256x256 pixels, in .jpg extension\n")


def get_training_dataset():
    should_end = False
    init_cli()
    is_pre_trained = 'yes'
    dataset_path, dataset_train, labels_train, names = None, None, None, None
    while not should_end:
        is_pre_trained = input("Do you wish to train with our pre trained network? (Yes/No): ")
        if is_pre_trained.lower() == "exit":
            should_end = True
            continue
        if is_pre_trained.lower() != 'yes' and is_pre_trained.lower() != 'no':
            print("No such option <:(")
            continue
        if is_pre_trained.lower() == 'no':
            dataset_path = input("Enter path to images")
            if not Path(dataset_path).exists():
                print("No such file <:(")
                dataset_path = None
            else:
                should_end = True
            # Labels can be an input to the program through a separate .csv
            # Right now we assume labels are in the filename
            # else:
            #     labels_path = input("Enter path to images labels (.csv)")
            #     print("IMPORTANT: To read labels in the correct order, it is assumed that the order of labels "
            #     "corresponds to the photos in alphabetical order")
            #     if not Path(labels_path).exists():
            #         print("No such file <:(")
            #         labels_path = None
            #     else:
            #         should_end = True
        else:
            should_end = True

    if is_pre_trained.lower() == 'yes':
        dataset_train, labels_train, names = load_images()
    if dataset_path is not None:
        dataset_train, labels_train, names = read_images(dataset_path)

    return dataset_train, labels_train, names


def is_pca():
    should_end = False
    pca = None
    while not should_end:
        kpca_or_pca = input("Do you wish to apply PCA or KPCA for data pre-processing?: ")
        if kpca_or_pca.lower() == "exit":
            should_end = True
            continue
        if kpca_or_pca.lower() != 'pca' and kpca_or_pca.lower() != 'kpca':
            print("No such opcion <:(")
            continue
        if kpca_or_pca.lower() == 'pca':
            pca = True
            should_end = True
        else:
            should_end = True
            pca = False
    return pca


def show_metrics():
    should_end = False
    show_metrics = None
    while not should_end:
        show_metrics = input("Do you wish to see testing metrics? (Yes/No): ")
        if show_metrics.lower() == 'exit':
            should_end = True
            continue
        if show_metrics.lower() != 'yes' and show_metrics.lower() != 'no':
            print("No such opcion <:(")
            continue
        if show_metrics.lower() == 'yes':
            should_end = True
            show_metrics = True
        else:
            should_end = True
            show_metrics = False
    return show_metrics

def read_images(path):
    file = Path(path)
    if not file.exists():
        print("No such file <:(")
        return None
    images = []
    labels_from_filename = []

    if not file.is_dir():
        images.append(np.array(Image.open(path), dtype=np.float))
        labels_from_filename.append(0)
        names_from_filename = [re.findall(r'[a-z]+', file.name)[0]]
    else:
        # Access all JPG files in directory
        DATA_PATH = str(path)+"/"
        allfiles = os.listdir(DATA_PATH)
        imlist = [filename for filename in allfiles if filename[-4:] in [".jpg", ".JPG"]]

        # Translate names in JPG files to labels
        names_from_filename = [re.findall(r'[a-z]+', filename)[0] for filename in allfiles if filename[-4:] in [".jpg", ".JPG"]]
        names_from_filename = list(dict.fromkeys(names_from_filename))
        names_from_filename.sort()

        for im in imlist:
            # images will be an array of (256,256,3) numpy arrays
            image = Image.open(DATA_PATH+im)
            width, height = image.size
            if width != 256 or height != 256:
                resizeImage(DATA_PATH+im)
                image = Image.open(DATA_PATH + im)
                print(f"Image {DATA_PATH + im} has been resized to 256x256")
            images.append(np.array(image, dtype=np.float))
            name = re.findall(r'[a-z]+', im)[0]
            labels_from_filename.append(names_from_filename.index(name))
    return np.array(images), np.array(labels_from_filename), np.array(names_from_filename)


def read_labels(path):
    file = Path(path)
    if not file.exists():
        print("No such file <:(")
        return None
    if file.is_dir():
        print("Labels should be a file, not a directory")
        return None
    return np.array(genfromtxt(file, delimiter=','))

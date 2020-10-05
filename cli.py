from numpy import genfromtxt
import numpy as np
from data_loading import load_images
from pathlib import Path
from PIL import Image
from pyfiglet import Figlet
import os
import re

def init_cli():
    f = Figlet(font='slant')
    print("Welcome!")
    print(f.renderText('FaceRecognition'))
    print("In order to quit write exit.")

def get_training_dataset():
    should_end = False
    init_cli()
    is_pre_trained = 'Yes'
    dataset_path, labels_path, dataset_train, labels_train, names = None, None, None, None, None
    while not should_end:
        is_pre_trained = input("Do you wish to train with our pre trained network? (Yes/No)")
        if is_pre_trained != 'Yes' and is_pre_trained != 'No':
            print("No such option <:(")
        if is_pre_trained == 'No':
            print("IMPORTANT: To read labels in the correct order, it is assumed that the order of labels "
                  "corresponds to the photos in alphabetical order")
            dataset_path = input("Enter path to images")
            if not Path(dataset_path).exists():
                print("No such file <:(")
                dataset_path = None
            else:
                labels_path = input("Enter path to images labels (.csv)")
                if not Path(labels_path).exists():
                    print("No such file <:(")
                    labels_path = None
                else:
                    should_end = True
        else:
            should_end = True

    if is_pre_trained == 'Yes':
        dataset_train, labels_train, names = load_images()
    if dataset_path is not None and labels_path is not None:
        dataset_train = read_images(dataset_path)
        labels_train = read_labels(labels_path)

    return dataset_train, labels_train, names


def read_images(path):
    file = Path(path)
    if not file.exists():
        print("No such file <:(")
        return None
    images = []
    labels_from_filename = []

    if not file.is_dir():
        images.append(np.array(Image.open(path), dtype=np.float))
        labels_from_filename.append(re.findall(r'[a-z]+', file.name))
        names_from_filename = [re.findall(r'[a-z]+', file.name)] # ache
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
            images.append(np.array(Image.open(DATA_PATH + im), dtype=np.float))
            name = re.findall(r'[a-z]+', im)[0]
            labels_from_filename.append(names_from_filename.index(name))
    return images, labels_from_filename, names_from_filename


def read_labels(path):
    file = Path(path)
    if not file.exists():
        print("No such file <:(")
        return None
    if file.is_dir():
        print("Labels should be a file, not a directory")
        return None
    return genfromtxt(file, delimiter=',')

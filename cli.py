import csv

from numpy import genfromtxt

from data_loading import load_images
from pathlib import Path
from glob import glob
from PIL import Image
from pyfiglet import Figlet

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
    if file.is_dir():
        filenames = glob(f"{path}/*.jpg")
        for f in filenames:
            images.append(Image.open(f))
    else:
        images.append(Image.open(path))
    return images


def read_labels(path):
    file = Path(path)
    if not file.exists():
        print("No such file <:(")
        return None
    if file.is_dir():
        print("Labels should be a file, not a directory")
        return None
    return genfromtxt(file, delimiter=',')

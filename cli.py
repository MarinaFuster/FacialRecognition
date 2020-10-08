import csv
from typing import Tuple, Union

import numpy as np
from numpy import genfromtxt

from data_loading import load_images
from pathlib import Path
from pyfiglet import Figlet


def init_cli():
    f = Figlet(font='slant')
    print("Welcome!")
    print(f.renderText('Face Mask'))
    print("In order to quit write exit.")
    print("IMPORTANT: In order to make things simple, labels should be in the filename.")


def get_images_path() -> str:
    dataset_path = input("Enter path to images: ")
    if not Path(dataset_path).exists():
        print("No such file <:(")
        return ""
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
    return dataset_path


def get_training_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    should_end = False
    init_cli()
    is_pre_trained = 'yes'
    dataset_path, dataset_train, labels_train, names = None, np.array(0), None, None
    while not should_end:
        is_pre_trained = input("Do you wish to train with our pre trained network? (Yes/No): ").lower()
        if is_pre_trained == "exit":
            should_end = True
            continue
        if is_pre_trained != 'yes' and is_pre_trained.lower() != 'no':
            print(f"{is_pre_trained} is not a valid response. Please write yes or no")
            continue
        if is_pre_trained.lower() == 'no':
            dataset_path = get_images_path()
            should_end = True
        else:
            should_end = True

    if is_pre_trained == 'yes' or dataset_path:
        dataset_train, labels_train, names = load_images(dataset_path)

    return dataset_train, labels_train, names


def is_pca():
    should_end = False
    pca = None
    while not should_end:
        kpca_or_pca = input("Do you wish to apply PCA or KPCA for training_data pre-processing?: ")
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


def read_labels(path):
    file = Path(path)
    if not file.exists():
        print("No such file <:(")
        return None
    if file.is_dir():
        print("Labels should be a file, not a directory")
        return None
    return np.array(genfromtxt(file, delimiter=','))

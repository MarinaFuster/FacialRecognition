from typing import Tuple, Union, Any, List

from classifier import Classifier
from cli import get_training_dataset
import numpy as np
from data_loading import load_images
from eigen import qr_eig_algorithm
from metrics import print_metrics
from sklearn.preprocessing import StandardScaler
from preprocessing import PreProcessing, KPCAPreprocessing, PCAPreprocessing


PRECISION = 0.95


def run_facial_recognition() -> None:
    """
    Main facial recognition program
    """
    # Initializing CLI interface and obtaining training dataset
    training_dataset, labels, names = get_training_dataset()
    ended: bool = True if training_dataset is None or labels is None or training_dataset.size == 0 or labels.size == 0 else False

    # Applying PCA or KPCA
    if not ended:
        is_pca: bool = should_run_pca()
        ended = True if is_pca is None or ended else False

    # Showing metrix
    if not ended:
        show_testing_metrics: bool = should_show_metrics()
        ended = True if show_testing_metrics is None or ended else False

    if ended:
        print("Exiting...")
        return

    # Training Classifier
    classifier = Classifier()
    preprocessing: PreProcessing
    pca_processing: np.ndarray

    preprocessing, pca_processing = train_with_svm(training_dataset, labels, names, classifier, is_pca)

    while not ended:
        path = input("Enter path to directory or path to image to test: ").lower()
        if path == 'exit':
            ended = True
        else:
            images, labels_test, names_test = load_images(path)
            if images.size == 0:
                continue
            if images.shape[0] == 0:
                print("There are no images to test")
                continue

            test_with_svm(images, classifier, preprocessing, pca_processing, show_testing_metrics,
                          labels_test, names_test=names_test, names=names)


def should_run_pca() -> bool:
    ended: bool = False

    while not ended:
        mode: str = input("Do you wish to apply PCA or KPCA for training  pre-processing?: ").lower()
        if mode == "exit":
            ended = True
        elif mode != "pca" and mode != "kpca":
            print("Invalid option")
            print(f"{mode} is not a valid pre-processing mode. Write PCA or KPCA.")
        elif mode == 'pca':
            return True
        else:
            return False

    return None


def should_show_metrics() -> bool:
    ended: bool = False

    while not ended:
        should_show: str = input("Do you wish to see testing metrics? (Yes/No): ").lower()
        if should_show == 'exit':
            ended = True
        elif should_show != 'yes' and should_show != 'no':
            print("Invalid option")
            print(f"{should_show} is not a valid option. Options are yes or no.")
        elif should_show == 'yes':
            return True
        else:
            return False

    return None


def train_with_svm(
        dataset: np.ndarray, labels: np.ndarray, names: np.ndarray, classifier: Classifier, is_pca: bool
) -> Tuple[PreProcessing, Union[KPCAPreprocessing, PCAPreprocessing]]:
    preprocessing = PreProcessing(dataset, dataset.shape[1], dataset.shape[2], dataset.shape[3])

    c_matrix: np.ndarray
    if is_pca:
        c_matrix = np.matmul(preprocessing.training_set, preprocessing.training_set.T)
    else:
        c_matrix = KPCAPreprocessing.get_kernel_pol_method(preprocessing.training_set)

    # Uses QR method to get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = qr_eig_algorithm(c_matrix)
    total = np.sum(np.abs(eigenvalues))

    acum = 0
    i = 0
    while acum < PRECISION:
        acum += eigenvalues[i] / total
        i = i + 1

    print(f"In order to win {round(acum, 4)} variance ratio we will use {i} eigenvectors")
    print("Training...")

    # Grab the first i eigenvectors
    eigenvectors = eigenvectors[:i]

    if is_pca:
        processing = PCAPreprocessing(preprocessing.training_set, preprocessing.avg_face, eigenvectors,
                                      dataset.shape[1], dataset.shape[2], dataset.shape[3], names, labels)
    else:
        processing = KPCAPreprocessing(preprocessing.training_set, preprocessing.avg_face, eigenvectors,
                                       dataset.shape[1], dataset.shape[2], dataset.shape[3], names,
                                       labels, c_matrix)
    # Feature scaling
    sc = StandardScaler()
    scaled_training_set = sc.fit_transform(processing.training_set)

    # Train classifier with default C and gamma values
    classifier.train_classifier(scaled_training_set, labels)
    classifier.save(preprocessing, processing)
    return preprocessing, processing


def preprocess_dataset(pca_processing: Union[KPCAPreprocessing, PCAPreprocessing],
                       preprocessing: PreProcessing, dataset: np.ndarray) -> Any:
    ret_list = []
    for data_i in dataset:
        stnd_img = preprocessing.regular_preprocess(data_i)
        ret_list.append(pca_processing.apply_method(stnd_img))

    return ret_list


def test_with_svm(dataset_test, classifier, preprocessing, pca_processing, show_testing_metrics,
                  labels, names_test, names) -> List:

    # Apply PCA/KPCA transformation to testing training_data
    dataset_test_pca = preprocess_dataset(pca_processing, preprocessing, dataset_test)
    labels_test_mapped_to_labels_train = []

    testing_with_training_dataset = True
    for label in labels:
        try:
            label_mapped = list(names).index(names_test[label])
        except:
            # If name is not in training dataset, then label is not mapped
            label_mapped = label
            # We can assume that user is not testing the dataset
            testing_with_training_dataset = False
            show_testing_metrics = False
        labels_test_mapped_to_labels_train.append(label_mapped)
    
    sc = StandardScaler()
    scaled_dataset_test_pca = sc.fit_transform(dataset_test_pca)

    
    # Test classifier
    y_pred = classifier.predict(scaled_dataset_test_pca)
    # classifier.save(preprocessing, pca_processing)

    # To obtain metrics
    print_metrics(y_pred, names, labels, labels_test_mapped_to_labels_train, names_test,
                  testing_with_training_dataset, show_testing_metrics)

    return [names[int(y_pred[i])] for i in range(len(y_pred))]


if __name__ == '__main__':
    run_facial_recognition()

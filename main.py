import numpy as np
import matplotlib.pyplot as plt
from metrics import print_metrics
from cli import get_training_dataset, read_images, is_pca, show_metrics
from eigen import calculate_eigenvectors
from preprocessing import PreProcessing, PCAPreprocessing, KPCAPreprocessing
from sklearn.decomposition import PCA

from classifier import Classifier


def preprocess_dataset(pca_processing, preprocessing, dataset):
    ret_list = []
    for data_i in dataset:
        stnd_img = preprocessing.regular_preprocess(data_i)
        ret_list.append(pca_processing.apply_pca(stnd_img))

    return ret_list


def train_with_svm(dataset_train, labels_train, classifier, is_pca, names):
    preprocessing = PreProcessing(dataset_train, dataset_train.shape[1], dataset_train.shape[2], dataset_train.shape[3])
    # Over this matrix we need to calculate eigenvectorss
    if is_pca:
        C_matrix = np.matmul(preprocessing.training_set, preprocessing.training_set.T)
    else:
        C_matrix = KPCAPreprocessing.rbf_kernel_pca(preprocessing.training_set)

    # From here ...
    pca_module = PCA(n_components=dataset_train.shape[0])
    pca_module.fit(C_matrix)

    accumulated = 0
    i = 0
    while accumulated < 0.95:
        accumulated = accumulated + pca_module.explained_variance_ratio_[i]
        i = i + 1
    print(f"In order to win {round(accumulated, 4)} variance ratio we will use {i} eigenvectors")

    # Grab the first i eigenvectors
    eigenvectors = calculate_eigenvectors(C_matrix)[:i+1]

    # Apply PCA transformation to training data
    pca_processing = PCAPreprocessing(preprocessing.training_set, preprocessing.avg_face, eigenvectors,
                                      dataset_train.shape[1], dataset_train.shape[2], dataset_train.shape[3], names, labels_train)

    # Train classifier with default C and gamma values
    classifier.train_classifier(pca_processing.training_set, labels_train)

    return preprocessing, pca_processing


def test_with_svm(dataset_test, classifier, preprocessing, pca_processing, show_testing_metrics,
                  labels_test, labels_train, names_test, names):
    # Apply PCA transformation to testing data
    dataset_test_pca = preprocess_dataset(pca_processing, preprocessing, dataset_test)

    labels_test_mapped_to_labels_train = []

    testing_with_training_dataset = True
    for label in labels_test:
        try:
            label_mapped = list(names).index(names_test[label])
        except:
            # If name is not in training dataset, then label is not mapped
            label_mapped = label
            # We can assume that user is not testing the dataset
            testing_with_training_dataset = False
            show_testing_metrics = False
        labels_test_mapped_to_labels_train.append(label_mapped)

    # Test classifier
    y_pred = classifier.predict(dataset_test_pca)

    # dataset_test = np.array(dataset_test_pca)
    # for i in range(dataset_test.shape[0]):
    #     pca_processing.reconstruct_image(dataset_test[i], names_test[labels_test[i]], names[y_pred[i]])

    # To obtain metrics
    print_metrics(y_pred, names, labels_test, labels_test_mapped_to_labels_train, names_test,
                  testing_with_training_dataset, show_testing_metrics)


if __name__ == '__main__':

    # Initializing CLI Interface and obtaining training dataset
    dataset_train, labels_train, names = get_training_dataset()
    should_end = True if dataset_train is None or labels_train is None else False

    # Applying PCA or KPCA
    is_pca = is_pca()
    should_end = True if is_pca is None else False

    # Showing metrics
    show_testing_metrics = show_metrics()
    should_end = True if show_metrics is None else False

    # Training classifier
    classifier = Classifier()
    preprocessing, pca_processing = None, None
    if not should_end:
        preprocessing, pca_processing = train_with_svm(dataset_train, labels_train, classifier, is_pca, names)
        print("Training done! Now you can try the face recognition (or write exit to exit)")

    # Testing classifier
    while not should_end:
        path = input("Enter path to images or path to image: ")
        if path.lower() == "exit":
            should_end = True
            continue
        images, labels_test, names_test = read_images(path)
        if images is None:
            continue
        if images.shape[0] == 0:
            print("There are no images to test.")
            continue
        test_with_svm(images, classifier, preprocessing, pca_processing, show_testing_metrics,
                      labels_test=labels_test, labels_train=labels_train, names_test=names_test, names=names)

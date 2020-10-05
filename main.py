import numpy as np
import matplotlib.pyplot as plt

from cli import get_training_dataset, read_images
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


def train_with_svm(dataset_train, labels_train, classifier):
    preprocessing = PreProcessing(dataset_train, dataset_train.shape[1], dataset_train.shape[2], dataset_train.shape[3])

    # Over this matrix we need to calculate eigenvectorss
    # C_matrix = np.matmul(preprocessing.training_set, preprocessing.training_set.T)
    K = KPCAPreprocessing.rbf_kernel_pca(preprocessing.training_set)
    C_matrix = K

    # From here ...
    pca_module = PCA(n_components=dataset_train.shape[0])
    pca_module.fit(C_matrix)

    accumulated = 0
    i = 0
    while accumulated < 0.95:
        accumulated = accumulated + pca_module.explained_variance_ratio_[i]
        i = i + 1
    print(f"In order to win {round(accumulated, 4)} variance ratio we will use {i} eigenvectors")

    eigenvectors = pca_module.components_[list(range(0, i))]
    # eigenvectors = calculate_eigenvectors(list(range(0, i)))
    # ... to here, must be replaced with eigenvectors calculated by eigen_calc

    # Apply PCA transformation to training data
    pca_processing = PCAPreprocessing(preprocessing.training_set, preprocessing.avg_face, eigenvectors,
                                      dataset_train.shape[1], dataset_train.shape[2], dataset_train.shape[3])

    # Train classifier with default C and gamma values
    classifier.train_classifier(pca_processing.training_set, labels_train)

    return preprocessing, pca_processing


def test_with_svm(dataset_test, classifier, preprocessing, pca_processing, labels_test=None, names=None):
    # Apply PCA transformation to testing data
    dataset_test_pca = preprocess_dataset(pca_processing, preprocessing, dataset_test)

    # Test classifier
    y_pred = classifier.predict(dataset_test_pca, labels_test)

    dataset_test = np.array(dataset_test_pca)
    for i in range(dataset_test.shape[0]):
        pca_processing.reconstruct_image(dataset_test[i], names[labels_test[i]], names[y_pred[i]])

    # To obtain a more readable output
    corrects = 0
    for i in range(len(y_pred)):
        if names is not None:
            print(f"Predicting label: {names[labels_test[i]]}. Face belongs to ... {names[int(y_pred[i])]}")
        if labels_test[i] == y_pred[i]:
            corrects = corrects + 1
    print(f"{corrects} out of {y_pred.shape[0]} were predicted properly")

    # Another way to show results
    # count = 0
    # processed_images = preprocess_dataset(pca_processing, preprocessing, images)
    # for processed_image in processed_images:
    #     y_pred = classifier.predict(processed_image)
    #     print(f"{filenames[count]} predicted as {names[y_pred]}")
    #     count += 1

if __name__ == '__main__':

    # Initializing CLI Interface and obtaining training dataset
    should_end = False
    dataset_train, labels_train, names = get_training_dataset()
    if dataset_train is None or labels_train is None:
        should_end = True

    # Training classifier
    classifier = Classifier()
    preprocessing, pca_processing = train_with_svm(dataset_train, labels_train, classifier)

    # Testing classifier
    print("Training done! Now you can try the face recognition (or write exit to exit)")
    while not should_end:
        path = input("Enter path to images")
        if path.lower() == "exit":
            should_end = True
            continue
        images = read_images(path)
        if images is None:
            continue
        test_with_svm(images, classifier, preprocessing, pca_processing, names=names)

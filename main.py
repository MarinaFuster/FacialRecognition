import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from preprocessing import PreProcessing, PCAPreprocessing, KPCAPreprocessing
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.model_selection import train_test_split

from classifier import Classifier
from neural_network import NNClassifier
from data_loading import load_images

def olivetti_faces():
    # Importing olivetti dataset
    faces = datasets.fetch_olivetti_faces()

    # Create feature and target set
    X = faces.images
    y = faces.target

    dataset_train, dataset_test, labels_train, labels_test = train_test_split(X, y, test_size=0.2)

    dataset_train = np.expand_dims(dataset_train, axis=-1)
    dataset_test = np.expand_dims(dataset_test, axis=-1)
    
    return dataset_train, dataset_test, labels_train, labels_test

def local_dataset():
    dataset, labels, names = load_images()
    # Shuffle in unison dataset and labels to obtain a trainning dataset with greater variance
    # dataset_combined = np.c_[dataset.reshape(len(dataset), -1), labels.reshape(len(labels), -1)]
    # np.random.shuffle(dataset_combined)
    # dataset = dataset_combined[:, :dataset.size//len(dataset)].reshape(dataset.shape)
    # labels = dataset_combined[:, dataset.size//len(dataset):].reshape(labels.shape)

    # Split dataset and labels in trainning and testing sets
    dataset_train, dataset_test, labels_train, labels_test = train_test_split(dataset, labels, test_size=0.2)
    
    return dataset_train, dataset_test, labels_train, labels_test, names

def train_with_svm(dataset_train, dataset_test, labels_train, labels_test, names=None):
    preprocessing = PreProcessing(dataset_train, dataset_train.shape[1], dataset_train.shape[2], dataset_train.shape[3])

    # Over this matrix we need to calculate eigenvectorss
    #C_matrix = np.matmul(preprocessing.training_set, preprocessing.training_set.T)
    K = KPCAPreprocessing.rbf_kernel_pca(preprocessing.training_set)
    C_matrix = K

    # From here ...
    pca_module = PCA(n_components=dataset_train.shape[0])
    pca_module.fit(C_matrix)

    accumulated = 0
    i = 0
    while accumulated < 0.95:
        accumulated = accumulated + pca_module.explained_variance_ratio_[i]
        i = i+1
    print(f"In order to win {round(accumulated,4)} variance ratio we will use {i} eigenvectors")

    eigenvectors = pca_module.components_[list(range(0,i))]
    # ... to here, must be replaced with eigenvectors calculated by eigen_calc

    # Apply PCA transformation to training data
    pca_processing = PCAPreprocessing(preprocessing.training_set, preprocessing.avg_face, eigenvectors, \
        dataset_train.shape[1], dataset_train.shape[2], dataset_train.shape[3])


    # Train classifier with default C and gamma values
    classifier = Classifier()
    classifier.train_classifier(pca_processing.training_set, labels_train)

    # Apply PCA transformation to testing data
    dataset_test_pca = []
    for data_i in dataset_test:
        stnd_img = preprocessing.regular_preprocess(data_i)
        dataset_test_pca.append(pca_processing.apply_pca(stnd_img))


    # Test classifier
    y_pred = classifier.predict(dataset_test_pca, labels_test)

    dataset_test = np.array(dataset_test_pca)
    for i in range(dataset_test.shape[0]):
        pca_processing.reconstruct_image(dataset_test[i], names[labels_test[i]], names[y_pred[i]])

    # To obtain a more readable output
    for i in range(len(y_pred)):
        print("Predicting: ", names[labels_test[i]], end ="")
        print(". Face belongs to ... ", names[int(y_pred[i])])

    corrects = 0
    for i in range(len(y_pred)):
        if labels_test[i] == y_pred[i]:
            corrects = corrects+1
    print(f"{corrects} out of {y_pred.shape[0]} were predicted properly")

def train_with_nn(dataset_train, dataset_test, labels_train, labels_test, names=None):
    preprocessing = PreProcessing(dataset_train, dataset_train.shape[1], dataset_train.shape[2], dataset_train.shape[3])

    # Over this matrix we need to calculate eigenvectorss
    C_matrix = np.matmul(preprocessing.training_set, preprocessing.training_set.T)
    # K = KPCAPreprocessing.rbf_kernel_pca(preprocessing.training_set)
    # C_matrix = K

    # From here ...
    pca_module = PCA(n_components=dataset_train.shape[0])
    pca_module.fit(C_matrix)

    accumulated = 0
    i = 0
    while accumulated < 0.95:
        accumulated = accumulated + pca_module.explained_variance_ratio_[i]
        i = i+1
    print(f"In order to win {round(accumulated,4)} variance ratio we will use {i} eigenvectors")

    eigenvectors = pca_module.components_[list(range(0,i))]
    # ... to here, must be replaced with eigenvectors calculated by eigen_calc

    # Apply PCA transformation to training data
    pca_processing = PCAPreprocessing(preprocessing.training_set, preprocessing.avg_face, eigenvectors, \
        dataset_train.shape[1], dataset_train.shape[2], dataset_train.shape[3])

    # Train classifier with default C and gamma values
    classifier = Classifier()
    classifier.train_classifier(pca_processing.training_set, labels_train)

    # Apply PCA transformation to testing data
    dataset_test_pca = []
    for data_i in dataset_test:
        stnd_img = preprocessing.regular_preprocess(data_i)
        dataset_test_pca.append(pca_processing.apply_pca(stnd_img))

    # Test classifier
    y_pred = classifier.predict(dataset_test_pca, labels_test)

    # Testing with another classifier to check if that's the problem
    nn_classifier = NNClassifier(pca_processing.training_set[0].shape, 5)
    nn_classifier.train(500, pca_processing.training_set, labels_train)

    y_pred = nn_classifier.predict(np.array(dataset_test_pca))

    corrects = 0
    for i in range(len(y_pred)):
        if labels_test[i] == y_pred[i]:
            corrects = corrects+1
    print(f"{corrects} out of {y_pred.shape[0]} were predicted properly")

if __name__ == '__main__':
    dataset_train, dataset_test, labels_train, labels_test, names = local_dataset()
    train_with_svm(dataset_train, dataset_test, labels_train, labels_test, names=names)
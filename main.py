import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from data_loading import load_images
from preprocessing import PreProcessing, PCAPreprocessing, KPCAPreprocessing
from sklearn.decomposition import PCA
from classifier import Classifier
from sklearn.model_selection import train_test_split

dataset, labels, paths, names = load_images()
# Shuffle in unison dataset and labels to obtain a trainning dataset with greater variance
# dataset_combined = np.c_[dataset.reshape(len(dataset), -1), labels.reshape(len(labels), -1)]
# np.random.shuffle(dataset_combined)
# dataset = dataset_combined[:, :dataset.size//len(dataset)].reshape(dataset.shape)
# labels = dataset_combined[:, dataset.size//len(dataset):].reshape(labels.shape)

# Split dataset and labels in trainning and testing sets
dataset_train, dataset_test, labels_train, labels_test = train_test_split(dataset, labels, test_size=0.02)

preprocessing = PreProcessing(dataset_train)

# Over this matrix we need to calculate eigenvectorss
C_matrix = np.matmul(preprocessing.training_set, preprocessing.training_set.T)
# K = KPCAPreprocessing.rbf_kernel_pca(X=preprocessing.training_set)
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
pca_processing = PCAPreprocessing(preprocessing.training_set, preprocessing.avg_face, eigenvectors)

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

# To obtain a more readable output
for i in range(len(y_pred)):
    print("Predicting: ", paths[i], end ="")
    print(". Face belongs to ... ", names[int(y_pred[i])])
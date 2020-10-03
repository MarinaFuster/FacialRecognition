import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from data_loading import load_images
from preprocessing import PreProcessing, PCAPreprocessing
from sklearn.decomposition import PCA
from classifier import Classifier
from sklearn.model_selection import train_test_split

dataset, labels, paths, names = load_images()

dataset_train, dataset_test, labels_train, labels_test = train_test_split(dataset, labels, test_size=0.2)

preprocessing = PreProcessing(dataset_train)

C_matrix = np.dot(preprocessing.training_set, preprocessing.training_set.T)
# print(C_matrix.shape)

pca_module = PCA(n_components=dataset_train.shape[0])
pca_module.fit(C_matrix)

accumulated = 0
i = 0
while accumulated < 0.9:
    accumulated = accumulated + pca_module.explained_variance_ratio_[i]
    i = i+1
print(f"In order to win {round(accumulated,4)} variance ratio we will use {i} eigenvectors")

eigenvectors = pca_module.components_[list(range(0,i))]

# Apply PCA transformation to training data
pca_processing = PCAPreprocessing(preprocessing.training_set, preprocessing.avg_face, eigenvectors)

# Train classifier with default C and gamma values
classifier = Classifier()
classifier.train_classifier(pca_processing.training_set, labels_train)

# Apply PCA transformation to testing data
# for data_i in dataset_test:

# Test classifier
y_pred = classifier.predict(dataset_test, labels_test)

# To obtain a more readable output
for i in range(len(y_pred)):
    print("Predicting: ", paths[i], end ="")
    print(". Face belongs to ... ", names[y_pred[i]])
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from data_loading import load_images
from preprocessing import PreProcessing, PCAPreprocessing
from sklearn.decomposition import PCA

dataset = load_images()

preprocessing = PreProcessing(dataset)

C_matrix = np.dot(preprocessing.training_set, preprocessing.training_set.T)
# print(C_matrix.shape)

pca_module = PCA(n_components=dataset.shape[0])
pca_module.fit(C_matrix)

accumulated = 0
i = 0
while accumulated < 0.9:
    accumulated = accumulated + pca_module.explained_variance_ratio_[i]
    i = i+1
print(f"In order to win {round(accumulated,4)} variance ratio we will use {i} eigenvectors")

eigenvectors = pca_module.components_[list(range(0,i))]

pca_processing = PCAPreprocessing(preprocessing.training_set, preprocessing.avg_face, eigenvectors)



import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from data_loading import load_images
from preprocessing import PreProcessing, PCAPreprocessing, KPCAPreprocessing
from sklearn.decomposition import PCA

dataset = load_images()

preprocessing = PreProcessing(dataset)

# Over this matrix we need to calculate eigenvectorss
C_matrix = np.matmul(preprocessing.training_set, preprocessing.training_set.T)
# K = KPCAPreprocessing.rbf_kernel_pca(X=preprocessing.training_set)
C_matrix = K

# From here ...
pca_module = PCA(n_components=dataset.shape[0])
pca_module.fit(C_matrix)

accumulated = 0
i = 0
while accumulated < 0.95:
    accumulated = accumulated + pca_module.explained_variance_ratio_[i]
    i = i+1
print(f"In order to win {round(accumulated,4)} variance ratio we will use {i} eigenvectors")

eigenvectors = pca_module.components_[list(range(0,i))]
# ... to here, must be replaced with eigenvectors calculated by eigen_calc

# Applies PCA to training set
pca_processing = PCAPreprocessing(preprocessing.training_set, preprocessing.avg_face, eigenvectors)



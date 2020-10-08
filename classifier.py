import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import decomposition
from sklearn import svm
from sklearn.model_selection import train_test_split
from joblib import dump, load
import numpy as np


class Classifier:

    # Gamma: defines how far the influence of a single training example reaches, with low values meaning ‘far’ and
    # high values meaning ‘close’. The gamma parameters can be seen as the inverse of the radius of influence of
    # samples selected by the model as support vectors. C: trades off correct classification of training examples
    # against maximization of the decision function’s margin. For larger values of C, a smaller margin will be
    # accepted if the decision function is better at classifying all training points correctly. A lower C will
    # encourage a larger margin, therefore a simpler decision function, at the cost of training accuracy.
    def __init__(self):
        self.clf = svm.LinearSVC(max_iter=100000)

    def train_classifier(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.clf.fit(x_train, y_train)

    def predict(self, x_test):
        return self.clf.predict(x_test)

    def save(self, preprocessing, pca_processing):
        dump(self.clf, 'models/classifier.jolib')
        dump(preprocessing, "models/preprocessing.jolib")
        dump(pca_processing, "models/pca_processing.jolib")


if __name__ == '__main__':

    # NOTE: this first part is for testing purposes as
    # we still have to connect the PCA transformation
    # prior to classifying

    # Importing olivetti dataset
    faces = datasets.fetch_olivetti_faces()

    # Create feature and target set
    X = faces.data
    y = faces.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    pca = decomposition.PCA(n_components=150, whiten=True)
    pca.fit(X_train)

    # Plot mean image
    plt.imshow(pca.mean_.reshape(faces.images[0].shape), cmap=plt.cm.gray)
    # plt.show()

    # Transform train and test datasets using trained PCA algorithm
    # Apply PCA transformation to training training_data
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # NOTE: this will stay as it is the main functionality
    # of this class
    classifier = Classifier()
    classifier.train_classifier(X_train_pca, y_train)
    y_pred = classifier.predict(X_test_pca)
    # print(y_test)
    # print(y_pred)

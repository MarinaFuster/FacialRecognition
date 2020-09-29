from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn import datasets
from sklearn import decomposition
import matplotlib.pyplot as plt
import scipy.io

class Classifier:

    def __init__(self, dataset):
        self.dataset = dataset
        self.X = dataset.data
        self.y = dataset.target
        self.target_names = dataset.target_names
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=split_dataset)

    def train_classifier(self, X_train, y_train, C=1.0, gamma=0.0001):
        self.clf = svm.SVC(C=C, gamma=gamma)
        self.clf.fit(X_train, y_train)

    def predict(self, X_test, y_test):
        y_pred = self.clf.predict(X_test)
        print(metrics.classification_report(y_test, y_pred))
        return y_pred


if __name__ == '__main__':
    # Importing olivetti dataset
    faces = datasets.fetch_olivetti_faces()
    faces.data.shape()

    # faces = scipy.io.loadmat('data/olivettifaces.mat')
    # todo; error: es type 'dict' por lo que no tiene los atributos data y target

    # Create feature and target set
    X = faces.data
    y = faces.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    pca = decomposition.PCA(n_components=150, whiten=True)
    pca.fit(X_train)

    # Plot mean image
    plt.imshow(pca.mean_.reshape(faces.images[0].shape), cmap=plt.cm.gray)

    # Transform train and test datasets using trained PCA algorithm
    # apply PCA transformation to training data
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # ahora somos nosotras
    classifier = Classifier(faces)
    classifier.train_classifier(X_train_pca, y_train)
    y_pred = classifier.predict(X_test_pca, y_test)
    print(y_pred)
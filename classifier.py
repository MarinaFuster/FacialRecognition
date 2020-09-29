import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import decomposition
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split


class Classifier:

    def __init__(self, C=5., gamma=0.001):
        self.clf = svm.SVC(C=C, gamma=gamma)

    def train_classifier(self, X_train_pca, y_train):
        self.clf.fit(X_train_pca, y_train)

    def predict(self, X_test, y_test):
        y_pred = self.clf.predict(X_test)
        print(metrics.classification_report(y_test, y_pred))
        return y_pred


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
    # Apply PCA transformation to training data
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # NOTE: this will stay as it is the main functionality
    # of this class
    classifier = Classifier()
    classifier.train_classifier(X_train_pca, y_train)
    y_pred = classifier.predict(X_test_pca, y_test)
    print(y_test)
    print(y_pred)

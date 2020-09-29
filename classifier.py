import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

# 1. Importing dataset
# todo: armar el dataset así
from sklearn import datasets

faces = datasets.fetch_olivetti_faces()
# faces.data.shape

# 2. Create X (features) and y (target) variabes from data and targets in the dataset. Check the shape.
X = faces.data
y = faces.target
# target_names = faces.target_names
# print(X.shape, y.shape)

# 3. Split the data into train and test sets using a 80/20 split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print(X_train.shape, X_test.shape)

# 4. Apply PCA to the training data
# todo: usar las funciones programadas por nosotros
from sklearn import decomposition

pca = decomposition.PCA(n_components=150, whiten=True)
pca.fit(X_train)

# 5. Compute the mean face
# One interesting part of PCA is that it computes the “mean” face, which can be interesting to
# examine. This can be computed with pca.mean_. This face will show you the mean for each dimension for all the
# images in the dataset. So it effectively shows you one MEAN face reflecting all the faces in the dataset.
plt.imshow(pca.mean_.reshape(faces.images[0].shape), cmap=plt.cm.gray)

# 6. Transform train and test datasets using trained PCA instance
# apply PCA transformation to training data
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# 7. Run an SVM Classifier
# Create and train an instance of SVM classifier
from sklearn import svm

clf = svm.SVC(C=5., gamma=0.001)
clf.fit(X_train_pca, y_train)

# 8. Pick up 30 images from the test data and predict their labels.
import numpy as np

fig = plt.figure(figsize=(16, 8))
for i in range(50):
    ax = fig.add_subplot(5, 10, i + 1, xticks=[], yticks=[])
    ax.imshow(X_test[i].reshape(faces.images[0].shape),
              cmap=plt.cm.gray)
    y_pred = clf.predict(X_test_pca[i, np.newaxis])[0]
    color = ('black' if y_pred == y_test[i] else 'red')
    ax.set_title(faces.target[y_pred],
                 fontsize='small', color=color)

# 9. Create a Classification Report
from sklearn import metrics

y_pred = clf.predict(X_test_pca)
print(metrics.classification_report(y_test, y_pred))

# El siguiente paso resume TODOS los anteriores! Podemos usar sólo esto
# 10. Create an scikit-learn Pipeline to chain together PCA and SVM
# Chain PCA and SVM to run above experiment in a single execution.
from sklearn.pipeline import Pipeline

clf = Pipeline([('pca', decomposition.PCA(n_components=150, whiten=True)),
                ('svm', svm.LinearSVC(C=1.0))])

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(metrics.confusion_matrix(y_pred, y_test))
print(metrics.classification_report(y_test, y_pred))


class Classifier:

    def __init__(self, dataset):
        self.dataset = dataset
        self.X = dataset.data
        self.y = dataset.target
        self.target_names = dataset.target_names

    # Training and applying PCA to dataset
    def pca_trainning(self, split=1.0):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=split)



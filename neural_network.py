import keras
from keras.optimizers import Adam

class NNClassifier():
    def __init__(self, input_shape, class_size):
        
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=input_shape),
            keras.layers.Dense(10),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Dense(5, activation='softmax')
        ])
        
        optimizer = Adam(lr=1e-3)
        model.compile(loss="mse", optimizer=optimizer)

        self.model = model

    def train(self, epochs, train_datastet, train_labels):
        H = self.model.fit(train_datastet, train_labels, epochs=epochs, verbose=0)
        self.H = H

    def predict(self, dataset):
        return self.model.predict(dataset)
    
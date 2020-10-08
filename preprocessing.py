import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from scipy.spatial.distance import pdist, squareform
from scipy import exp

class PreProcessing:
    def __init__(self, dataset, height, width, channels):
        self.height: int = height
        self.width: int = width
        self.channels: int = channels
        self.dataset: np.ndarray = dataset
        self.avg_face = self.__get_avg_face(dataset)
        self.training_set = self.__get_training_set()
        if self.channels == 3:
            self.__save_avg_image()

    """ Calculates average face from dataset """
    def __get_avg_face(self, dataset: np.ndarray) -> np.ndarray:
        # Assuming all images are the same size, get dimensions of first image
        dataset_size: int = len(self.dataset)

        # Create a np array of floats to store the average (assume RGB images)
        self.avg_face = np.zeros((self.height, self.width, self.channels), dtype=np.float)

        # Build up average pixel intensities, casting each image as an array of floats
        for face in self.dataset:
            self.avg_face = self.avg_face + face/dataset_size
        
        # Round values in array and cast as 8-bit integer
        return np.array(np.round(self.avg_face), dtype=np.uint8)

    """ Save avg image to eigenfaces folder """
    def __save_avg_image(self):
        # Generate, save and preview final image
        out = Image.fromarray(self.avg_face, mode="RGB")
        out.save("eigenfaces/average.png")
    
    """ Subtracts avg face from dataset """
    def __standardize_dataset(self) -> np.ndarray:
        # standardize dataset to be centered at 0 and values range from [-1,1]
        return self.dataset-self.avg_face
    
    """ Flattens all images to (256,256,3) to 256*256*3 dimensions """
    def __flatten_dataset(self, standarized_set: np.ndarray) -> np.ndarray:
        return np.array([xi.flatten() for xi in standarized_set])
    
    """ Preprocesses dataset that will be used for training set 
        Applies both standardize and flatten methods """
    def __get_training_set(self) -> np.ndarray:
        standardized: np.ndarray = self.__standardize_dataset()
        flatten: np.ndarray = self.__flatten_dataset(standardized)
        return flatten
    
    """ Every time we want to test an image, we MUST call this method before applying PCA preprocessing """
    # This must be a (256, 256, 3) np array
    def regular_preprocess(self, image):
        if image.shape != self.avg_face.shape:
            print(f"Your image shape should be the same as {self.avg_face.shape}")
            return
        
        # standardize image to be centered and values will range from [-1,1]
        standardize = (image - self.avg_face)/255.0
        return standardize.flatten()


class PCAPreprocessing:

    def __init__(self, dataset, avg_face, eigenvectors, height, width, channels, names, labels_train):
        self.height: np.int64 = height
        self.width: np.int64 = width
        self.channels: np.int64 = channels
        self.avg_face: np.ndarray = avg_face
        self.eigenvectors: np.ndarray = eigenvectors
        self.eigenfaces: np.ndarray = self.__get_eigenfaces(dataset)
        self.training_set: np.ndarray = self.__apply_pca(dataset)
        self.names = names
        self.labels_train = labels_train
        if self.channels == 3:
            self.__save_eigenfaces(3)
            self.__save_dataset_projections()

    """ Applies PCA to dataset and returns training set for classifier """
    def __apply_pca(self, dataset: np.ndarray) -> np.ndarray:
        training_set = []
        for image in dataset:
            coords = []
            for face in self.eigenfaces:
                coords.append(np.dot(face, image))
            training_set.append(np.array(coords))
        return np.array(training_set)

    """ Calculates eigenfaces using dataset and eigenvector that represent some percentage of information.
        Eigenfaces work as a basis for all images """
    def __get_eigenfaces(self, dataset: np.ndarray) -> np.ndarray:
        eigenfaces = []
        for vector in self.eigenvectors:
            eigenface = np.zeros(dataset[0].shape)
            for i in range(vector.shape[0]):
                eigenface = eigenface + vector[i]*dataset[i]
            eigenfaces.append(eigenface)

        return np.array(eigenfaces)
    
    """ Eigenfaces are saved in the `eigenfaces/` directory"""
    def __save_eigenfaces(self, n) -> None:
        for i in range(n):
            reshaped = np.reshape(self.eigenfaces[i], (self.height, self.width, self.channels)) + self.avg_face
            reshaped = reshaped/255
            plt.imshow(reshaped)
            plt.savefig(f"eigenfaces/eigenface_{i}.png")
            plt.clf()
    
    """ Plots all elements from training set onto the first three eigenfaces dimensional space """
    def __save_dataset_projections(self) -> None:
        x_projection = self.training_set[:, 0]
        y_projection = self.training_set[:, 1]
        z_projection = self.training_set[:, 2]

        ax = plt.axes(projection="3d")

        for i in range(len(x_projection)):
            ax.scatter(x_projection[i], y_projection[i], z_projection[i], color='b')
            ax.text(x_projection[i], y_projection[i], z_projection[i],
                    '%s' % (self.names[self.labels_train[i]]), size=7, zorder=1, color='k')

        ax.set_xlabel("First component")
        ax.set_ylabel("Second component")
        ax.set_zlabel("Third component")

        plt.title("Projection of images on eigenface dimensional space")
        plt.savefig("eigenfaces/dataset_projection.png")
        plt.clf()
    
    """ After an image was regular preprocessed, we call this method to apply PCA.
        This method returns coords of the test image on defined eigenfaces """
    def apply_method(self, image) -> np.ndarray:
        if image.shape[0] != self.height*self.width*self.channels:
            print("Your image shape should be (256,256,3) flatten")
            return None
        coords = []
        for face in self.eigenfaces:
            coords.append(np.dot(face, image))
        return np.array(coords)
    
    def reconstruct_image(self, pca_coords, label, predicted_label):
        flatten = np.zeros(self.height * self.width * self.channels)
        for i in range(self.eigenfaces.shape[0]):
            flatten += pca_coords[i]*self.eigenfaces[i]
        reshaped = np.reshape(flatten, (self.height, self.width, self.channels)) + self.avg_face
        reshaped = reshaped
        plt.title(f"Label: {label}  -  Predicted Label: {predicted_label}")
        plt.imshow(reshaped)
        plt.show()
        plt.clf()

class KPCAPreprocessing:

    def __init__(self, dataset, avg_face, eigenvectors, height, width,
                 channels, names, labels_train, K):
        self.M: np.int64 = K.shape[0]
        print(f"M value is {self.M}")
        self.height: np.int64 = height
        self.width: np.int64 = width
        self.channels: np.int64 = channels
        self.avg_face: np.ndarray = avg_face
        
        self.eigenvectors = eigenvectors
        
        # about current dataset
        self.dataset = dataset
        self.names = names
        self.labels_train = labels_train
        self.dataset = dataset
        self.K = K
        self.training_set = self.__apply_kpca()
        if self.channels == 3:
            # self.__save_eigenfaces()
            self.__save_dataset_projections()

    """ Applies PCA to dataset and returns training set for classifier """
    def __apply_kpca(self):
        proyections = np.dot(self.K.T, self.eigenvectors.T)
        print(f"Proyections of dataset onto KPCA has shape {proyections.shape}")
        
        return proyections # applies kpca to dataset

    """ Plots all elements from training set onto the first three eigenfaces dimensional space """
    def __save_dataset_projections(self) -> None:
        x_projection = self.training_set[:, 0]
        y_projection = self.training_set[:, 1]
        z_projection = self.training_set[:, 2]

        ax = plt.axes(projection="3d")

        for i in range(len(x_projection)):
            ax.scatter(x_projection[i], y_projection[i], z_projection[i], color='b')
            ax.text(x_projection[i], y_projection[i], z_projection[i],
                    '%s' % (self.names[self.labels_train[i]]), size=7, zorder=1, color='k')

        plt.title("Projection of images on eigenface dimensional space")
        plt.savefig("eigenfaces/dataset_projection.png")
        plt.clf()
    
    """ After an image was regular preprocessed, we call this method to apply KPCA.
        This method returns coords of the test image on defined eigenfaces """
    def apply_method(self, image, DEGREE=2):
        ones_test = np.ones([1, self.M]) / self.M
        
        K_test = (np.dot([image], self.dataset.T) + 1) ** DEGREE
        
        ones_test_dot_K = np.dot(ones_test, self.K)
        
        ones = np.ones([self.M, self.M]) / self.M # weak point ?
        
        K_test = K_test - ones_test_dot_K - np.dot(K_test, ones) + np.dot(ones_test_dot_K, ones)
        
        proyection = np.dot(K_test, self.eigenvectors.T)
        
        return proyection[0] # apply pca to one image TODO complete

    @staticmethod
    def get_kernel_pol_method(images, DEGREE=2):
        M = images.shape[0]
        
        K = (np.dot(images, images.T) + 1) ** DEGREE
        ones = np.ones([M, M]) / M
        K_dot_ones = np.dot(K, ones)
        K = K - np.dot(ones, K) - K_dot_ones + np.dot(ones, K_dot_ones)
        
        return K

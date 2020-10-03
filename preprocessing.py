import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh

class PreProcessing():
    def __init__(self, dataset):
        self.dataset = dataset
        self.avg_face = self.__get_avg_face(dataset)
        self.training_set = self.__get_training_set(dataset, self.avg_face)
        self.__save_avg_image()

    """ Calcluates avg face from dataset """
    def __get_avg_face(self, dataset):
        # Assuming all images are the same size, get dimensions of first image
        w,h=self.dataset[0].shape[0], self.dataset[0].shape[1]
        N=len(self.dataset)

        # Create a np array of floats to store the average (assume RGB images)
        self.avg_face=np.zeros((h,w,3),np.float)

        # Build up average pixel intensities, casting each image as an array of floats
        for face in self.dataset:
            self.avg_face = self.avg_face + face/N
        
        # Round values in array and cast as 8-bit integer
        return np.array(np.round(self.avg_face),dtype=np.uint8)

    """ Save avg image to eigenfaces folder """
    def __save_avg_image(self):
        # Generate, save and preview final image
        out=Image.fromarray(self.avg_face,mode="RGB")
        out.save("eigenfaces/average.png")
    
    """ Substracts avg face from dataset and values range from [-1,1] """
    def __standardize_dataset(self, dataset, avg_face):
        # standardize dataset to be centered at 0 and values range from [-1,1]
        return (dataset - avg_face)/255.0
    
    """ Flattens all images to (256,256,3) to 256*256*3 dimensions """
    def __flatten_dataset(self, dataset):
        return np.array([xi.flatten() for xi in dataset])
    
    """ Preprocesses dataset that will be used for training set 
        Applies both standardize and flatten methods """
    def __get_training_set(self, dataset, avg_face):
        standardized = self.__standardize_dataset(dataset, self.avg_face)
        flatten = self.__flatten_dataset(standardized)
        return flatten
    
    """ Every time we want to test an image, we MUST call this method before applying PCA preprocessing """
    # This must be a (256, 256, 3) np array
    def regular_preprocess(self, image):
        if image.shape != self.avg_face.shape:
            print("Your image shape should be (256,256,3)")
            return
        
        # standardize image to be centered and values will range from [-1,1]
        standardize = (image - self.avg_face)/255.0
        return standardize.flatten()

class PCAPreprocessing():
    def __init__(self, dataset, avg_face, eigenvectors):
        self.avg_face = avg_face
        self.eigenvectors = eigenvectors
        self.eigenfaces = self.__get_eigenfaces(dataset, eigenvectors)
        self.training_set = self.__apply_pca(dataset)
        self.__save_eigenfaces()
        self.__save_dataset_projections()

    """ Applies PCA to dataset and returns training set for classifier """
    def __apply_pca(self, dataset):
        training_set = []
        for im in dataset:
            coords = []
            for face in self.eigenfaces:
                coords.append(np.dot(face.T, im))
            training_set.append(np.array(coords))
        return np.array(training_set)
    
    """ Calculates eigenfaces using dataset and eigenvector that represent some percentage of information.
        Eigenfaces work as a basis for all images """
    def __get_eigenfaces(self, dataset, eigenvectors):
        eigenfaces = []
        for vector in eigenvectors:
            eigenface = np.zeros(dataset[0].shape)
            for i in range(vector.shape[0]):
                eigenface = eigenface + vector[i]*dataset[i]
            eigenfaces.append(eigenface)
        return np.array(eigenfaces)
    
    """ Eigenfaces are saved in eigenfaces/"""
    def __save_eigenfaces(self):
        for i in range(self.eigenfaces.shape[0]):
            reshaped = np.reshape(self.eigenfaces[i], (256, 256, 3))*255 + self.avg_face
            reshaped = reshaped/255
            plt.imshow(reshaped)
            plt.savefig(f"eigenfaces/eigenface_{i}.png")
            plt.clf()
    
    """ Plots all elements from training set onto the first three eigenfaces dimensional space """
    def __save_dataset_projections(self):
        X = self.training_set[:,0]
        Y = self.training_set[:,1]
        Z = self.training_set[:,2]
        
        ax = plt.axes(projection = "3d")
        ax.scatter3D(X,Y,Z)
        
        plt.title("Projection of images on eigenface dimensional space")
        plt.savefig("eigenfaces/dataset_projection.png")
    
    """ After an image was regular preprocessed, we call this method to apply PCA.
        This method returns coords of the test image on defined eigenfaces """
    def apply_pca(self, image):
        if image.shape[0] != 256*256*3:
            print("Your image shape should be (256,256,3) flatten")
            return
        coords = []
        for face in self.eigenfaces:
            coords.append(np.dot(face.T, image))
        return np.array(coords)

class KPCAPreprocessing():
    def rbf_kernel_pca(self, X, gamma, n_components):
        """
        RBF kernel PCA implementation.    
        Parameters
        ------------
        X: {NumPy ndarray}, shape = [n_examples, n_features]  
        gamma: float
            Tuning parameter of the RBF kernel    
        n_components: int
            Number of principal components to return    
        Returns
        ------------
        X_pc: {NumPy ndarray}, shape = [n_examples, k_features]
            Projected dataset   
        """
        # Calculate pairwise squared Euclidean distances
        # in the MxN dimensional dataset.
        sq_dists = pdist(X, 'sqeuclidean')    
        # Convert pairwise distances into a square matrix.
        mat_sq_dists = squareform(sq_dists)    
        # Compute the symmetric kernel matrix.
        K = exp(-gamma * mat_sq_dists)    
        # Center the kernel matrix.
        N = K.shape[0]
        one_n = np.ones((N,N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)    
        # Obtaining eigenpairs from the centered kernel matrix
        # scipy.linalg.eigh returns them in ascending order
        eigvals, eigvecs = eigh(K)
        eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]    
        # Collect the top k eigenvectors (projected examples)
        X_pc = np.column_stack([eigvecs[:, i]
                            for i in range(n_components)])    
        return X_pc
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh

class PreProcessing():
    def __init__(self, dataset, h, w, d):
        self.h = h
        self.w = w
        self.d = d
        self.dataset = dataset
        self.avg_face = self.__get_avg_face(dataset)
        self.training_set = self.__get_training_set(dataset, self.avg_face)
        if self.d == 3:
            self.__save_avg_image()

    """ Calcluates avg face from dataset """
    def __get_avg_face(self, dataset):
        # Assuming all images are the same size, get dimensions of first image
        N=len(self.dataset)

        # Create a np array of floats to store the average (assume RGB images)
        self.avg_face=np.zeros((self.h,self.w,self.d),np.float)

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
            print(f"Your image shape should be the same as {self.avg_face.shape}")
            return
        
        # standardize image to be centered and values will range from [-1,1]
        standardize = (image - self.avg_face)/255.0
        return standardize.flatten()

class PCAPreprocessing():
    def __init__(self, dataset, avg_face, eigenvectors, h, w, d, names, labels_train):
        self.h = h
        self.w = w
        self.d = d
        self.avg_face = avg_face
        self.eigenvectors = eigenvectors
        self.eigenfaces = self.__get_eigenfaces(dataset, eigenvectors)
        self.training_set = self.__apply_pca(dataset)
        self.names = names
        self.labels_train = labels_train
        if self.d == 3:
            self.__save_eigenfaces()
            self.__save_dataset_projections()

    """ Applies PCA to dataset and returns training set for classifier """
    def __apply_pca(self, dataset):
        training_set = []
        for im in dataset:
            coords = []
            for face in self.eigenfaces:
                coords.append(np.dot(face, im)/np.linalg.norm(face))
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
            eigenfaces.append(eigenface/np.linalg.norm(eigenface))
        return np.array(eigenfaces)
    
    """ Eigenfaces are saved in eigenfaces/"""
    def __save_eigenfaces(self):
        for i in range(self.eigenfaces.shape[0]):
            reshaped = np.reshape(self.eigenfaces[i], (self.h, self.w, self.d))*255 + self.avg_face
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

        for i in range(len(X)):
            ax.scatter(X[i],Y[i],Z[i], color= 'b')
            ax.text(X[i],Y[i],Z[i], '%s'%(self.names[self.labels_train[i]]),size=7,zorder= 1, color='k')
        
        # ax = plt.axes(projection = "3d")
        # ax.scatter3D(X,Y,Z)
    

        ax.set_xlabel("First component")
        ax.set_ylabel("Second component")
        ax.set_zlabel("Third component")

        
        plt.title("Projection of images on eigenface dimensional space")
        plt.savefig("eigenfaces/dataset_projection.png")
        plt.clf()
    
    """ After an image was regular preprocessed, we call this method to apply PCA.
        This method returns coords of the test image on defined eigenfaces """
    def apply_pca(self, image):
        if image.shape[0] != self.h*self.w*self.d:
            print("Your image shape should be (256,256,3) flatten")
            return
        coords = []
        for face in self.eigenfaces:
            coords.append(np.dot(face, image)/np.linalg.norm(face))
        return np.array(coords)
    
    def reconstruct_image(self, pca_coords, label, predicted_label):
        flatten = np.zeros(self.h*self.w*self.d)
        for i in range(self.eigenfaces.shape[0]):
            flatten += pca_coords[i]*self.eigenfaces[i]
        reshaped = np.reshape(flatten, (self.h, self.w, self.d))*255 + self.avg_face
        reshaped = reshaped/255
        plt.title(f"Label: {label}  -  Predicted Label: {predicted_label}")
        plt.imshow(reshaped)
        plt.show()
        plt.clf()
            
class KPCAPreprocessing():
    @staticmethod
    def rbf_kernel_pca(dataset, gamma=0.000001):

        # Calculate pairwise squared Euclidean distances
        # in the MxN dimensional dataset.
        sq_dists = pdist(dataset, 'sqeuclidean')    
        # Convert pairwise distances into a square matrix.
        mat_sq_dists = squareform(sq_dists)    
        # Compute the symmetric kernel matrix.
        K = exp(-gamma * mat_sq_dists)    
        # Center the kernel matrix.
        N = K.shape[0]
        one_n = np.ones((N,N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
  
        return K
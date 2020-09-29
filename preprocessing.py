import numpy as np
from PIL import Image


# IMPORTANT
# @field dataset: original images dataset to train NN for classification
# @field avg_face: avg_face calculated from dataset in order to standardize images (centered at 0)
# @field trainning_set: all images are now standardized (centered at 0) and their values range from -1 to 1
class PreProcessing():
    def __init__(self, dataset):
        self.dataset = dataset
        self.calculate_avg_face()
        self.standardize_dataset()

    def calculate_avg_face(self, dataset):
        # Assuming all images are the same size, get dimensions of first image
        w,h=self.dataset[0].shape[0], self.dataset[0].shape[1]
        N=len(self.dataset)

        # Create a np array of floats to store the average (assume RGB images)
        self.avg_face=np.zeros((h,w,3),np.float)

        # Build up average pixel intensities, casting each image as an array of floats
        for face in self.dataset:
            self.avg_face = self.avg_face + face/N
        
        # Round values in array and cast as 8-bit integer
        self.avg_face=np.array(np.round(self.avg_face),dtype=np.uint8)

    # This method allows you to save the avg image on your current folder
    def save_avg_image():
        # Generate, save and preview final image
        out=Image.fromarray(self.avg_face,mode="RGB")
        out.save("Average.png")
        out.show()
    
    def standardize_dataset():
        # standardize dataset to be centered at 0 and values range from [-1,1]
        self.trainning_set = (self.dataset - self.avg_face)/255.0

    # This must be a (255, 255, 3) np array
    def standardize_input_image(image):
        if image.shape != self.avg_face.shape:
            print("Your image shape should be (255,255,3)")
            return
        
        # standardize image to be centered and values will range from [-1,1]
        return (image - self.avg_face)/255.0
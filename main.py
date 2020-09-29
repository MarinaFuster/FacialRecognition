from data_loading import load_images
from preprocessing import PreProcessing


dataset = load_images()

preprocessing = PreProcessing(dataset)
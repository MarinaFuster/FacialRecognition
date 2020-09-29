import numpy as np
import os
from PIL import Image

# IMPORTANT
# in this folder, only already functional faces will be stored.
# If you need something like face detection, DO NOT store those images here
DATA_PATH = "data/"

def load_images():
    # Access all JPG files in directory
    allfiles=os.listdir(DATA_PATH)
    imlist=[filename for filename in allfiles if filename[-4:] in [".jpg",".JPG"]]

    images = []
    for im in imlist:
        # images will be an array of (256,256,3) numpy arrays
        images.append(np.array(Image.open(DATA_PATH + im),dtype=np.float))

    return np.array(images)

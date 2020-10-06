import numpy as np
import os
from PIL import Image
from image_resize import resizeImage
import re

# IMPORTANT
# in this folder, only already functional faces will be stored.
# If you need something like face detection, DO NOT store those images here
DATA_PATH = "training_data/"


def load_images():
    # Access all JPG files in directory
    allfiles = os.listdir(DATA_PATH)
    imlist = [filename for filename in allfiles if filename[-4:] in [".jpg", ".JPG"]]

    # Translate names in JPG files to labels
    names = [re.findall(r'[a-z]+', filename)[0] for filename in allfiles if filename[-4:] in [".jpg", ".JPG"]]
    names = list(dict.fromkeys(names))
    names.sort()

    images = []
    labels = []
    for im in imlist:
        image = Image.open(DATA_PATH+im)
        width, height = image.size
        if width != 256 or height != 256:
            resizeImage(DATA_PATH+im)
            image = Image.open(DATA_PATH + im)
            print(f"Image {DATA_PATH + im} has been resized to 256x256")
        # images will be an array of (256,256,3) numpy arrays
        images.append(np.array(image, dtype=np.float))
        # We assume that the first part of the image path that has only [a-z] characters is the face name
        name = re.findall(r'[a-z]+', im)[0]
        labels.append(names.index(name))

    return np.array(images), np.array(labels), np.array(names)

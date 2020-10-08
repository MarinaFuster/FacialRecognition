from pathlib import Path
from typing import Tuple

import numpy as np
import os
from PIL import Image
import re

# IMPORTANT
# in this folder, only already functional faces will be stored.
# If you need something like face detection, DO NOT store those images here
DATA_PATH = "training_data/"
WIDTH, HEIGHT = 256, 256


def load_images(path: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    images = []
    labels = []

    if path:
        file = Path(path)
        if not file.exists():
            print(f"{path} is not a valid path")
            return None, None, None
        if not file.is_dir():
            images.append(np.array(Image.open(path).resize((WIDTH, HEIGHT), Image.ANTIALIAS), dtype=np.float))
            labels.append(0)
            names = [re.findall(r'[a-z]+', file.name)[0]]
            return np.array(images), np.array(labels), np.array(names)

    path = f"{path}/" if path else DATA_PATH
    # Access all JPG files in directory
    if not Path(path).exists():
        print("Directory does not exist")
        return None, None, None

    files = os.listdir(path)
    image_files = [filename for filename in files if filename[-4:] in [".jpg", ".JPG"]]

    # Translate names in JPG files to labels
    names = [re.findall(r'[a-z]+', filename)[0] for filename in files if filename[-4:] in [".jpg", ".JPG"]]
    names = list(dict.fromkeys(names))
    names.sort()

    for im in image_files:
        image_path = f"{path}{im}"
        image = Image.open(image_path)
        width, height = image.size
        if width != WIDTH or height != HEIGHT:
            image = image.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
            print(f"Image {image_path} has been resized to {WIDTH}x{HEIGHT}")
        # images will be an array of (256,256,3) numpy arrays
        images.append(np.array(image, dtype=np.float))
        # We assume that the first part of the image path that has only [a-z] characters is the face name
        name = re.findall(r'[a-z]+', im)[0]
        labels.append(names.index(name))

    return np.array(images), np.array(labels), np.array(names)

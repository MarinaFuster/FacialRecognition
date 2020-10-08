from PIL import Image


def resize_image(filepath):
    pixels = 256
    img = Image.open(filepath)
    img = img.resize((pixels,pixels), Image.ANTIALIAS)
    img.save(filepath)

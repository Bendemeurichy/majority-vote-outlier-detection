import tifffile as tiff
import numpy as np


def handle_tiff(input_path) -> np.ndarray:
    """Preprocess a tiff file.
    Only layers 1,2 and 3 are usefull of the 8 layers.
    return the images as 1 layer that is the mean of the 3 layers.

    Args:
        input_path (str): Path to the tiff file.

    Returns:
        np.ndarray: Preprocessed tiff file.
    """
    img = tiff.imread(input_path)
    img = img[1:4]
    img = np.mean(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img


def flatten_image(image: np.ndarray) -> np.ndarray:
    """
    Flatten the image to a 1D array
    :param image: image to flatten as output from handle_tiff() (height x width)
    :return: 1D array of image
    """
    img = image.squeeze()
    return img.flatten()

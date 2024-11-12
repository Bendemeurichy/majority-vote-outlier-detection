import tifffile as tiff
import numpy as np
import torch



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
    return img

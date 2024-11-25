import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tifffile as tiff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils import load_csv
from preprocessing import tiff_handling

DEBUG = False


def filter_single_cells(data: pd.DataFrame) -> pd.DataFrame:
    """Filter the data to only include single cells.

    Args:
        data (pd.DataFrame): Data to filter.

    Returns:
        pd.DataFrame: Filtered data.
    """
    return data[data["classification"] == 1]


def VGG_features(dataset: pd.DataFrame, target_size: tuple = (60, 80)) -> pd.DataFrame:
    """Extracts features from the dataset(images) using a pretrained VGG16 model.

    Args:
        dataset (pd.DataFrame): Dataframe with the last 3 columns of the features.csv file.
        target_size (tuple, optional): Target size(height, width) for the images. Defaults to (60, 80).

    Returns:
        pd.DataFrame: Dataframe with the extracted features([]).
    """
    print("--------------------- Extracting features using VGG16 ---------------------")

    model = VGG16(weights="imagenet", include_top=False)

    all_features = []

    print(f"Extracting features from {len(dataset)} images") if DEBUG else None

    # Process the images
    num_images = len(dataset)
    for i in range(0, num_images):
        current_image = dataset["file_names"].iloc[i]  # Use iloc for indexing

        print(f"Loading image: {current_image}") if DEBUG else None

        img = tiff.imread(current_image)
        img_selected = img[[1, 2, 3], :, :]  # Resulting shape: (3, H, W)

        # Transpose to (H, W, 3) for compatibility with TensorFlow's resize
        img_rgb = np.transpose(img_selected, (1, 2, 0))

        # Resize the floating-point image using TensorFlow
        img_rgb_resized = tf.image.resize(img_rgb, target_size, method="bilinear")

        # Convert the resized image to a NumPy array, add batch dimension, and preprocess for VGG16
        x = np.expand_dims(img_rgb_resized.numpy(), axis=0)  # Add batch dimension
        x = preprocess_input(x)  # VGG16 expects preprocessed input

        # Extract features
        current_features = model.predict(x)
        flattened_features = current_features.flatten()

        # Append to the list of features
        all_features.append(flattened_features)

    # Convert list of features into a DataFrame
    features_df = pd.DataFrame(all_features)

    # Concatenate the labels from the dataset
    output_df = pd.concat(
        [features_df, dataset[["classification"]].reset_index(drop=True)], axis=1
    )

    print(f"Extracted features shape: {output_df.shape}") if DEBUG else None
    print(output_df) if DEBUG else None

    return output_df


def append_pseudo_negative_data(
    data: pd.DataFrame, fraction: float = 1.0
) -> pd.DataFrame:
    """Append pseudo-negative data to the dataset.

    Args:
        data (pd.DataFrame): Data to append pseudo-negative data to.
        num_samples (float, optional): precentage of pseudo-negative samples to generate. Defaults to 1.0.

    Returns:
        pd.DataFrame: Data with appended pseudo-negative data.
    """
    print("--------------------- Appending pseudo-negative data ---------------------")

    # Set default for num_samples to match the number of rows in the original dataset
    num_samples = int(len(data) * fraction)

    # Exclude the label column to focus on the feature columns only
    feature_columns = data.columns.drop("classification")
    features = data[feature_columns]

    # Calculate the mean (mu) and standard deviation (sigma) for each feature
    mu = features.mean().to_numpy(dtype=np.float64)  # Shape (D,)
    sigma = features.std().to_numpy(dtype=np.float64)  # Shape (D,)

    # Generate pseudo-negative data from a Gaussian distribution
    pseudo_negative_data = np.random.normal(
        loc=mu, scale=sigma, size=(num_samples, len(feature_columns))
    )

    # Create a DataFrame for the pseudo-negative data and assign a label
    pseudo_negative_df = pd.DataFrame(pseudo_negative_data, columns=feature_columns)
    pseudo_negative_df["classification"] = 2  # Label for pseudo-negative data

    # Concatenate the original data with the pseudo-negative data
    augmented_data = pd.concat([data, pseudo_negative_df], ignore_index=True)

    print(f"Appended {num_samples} pseudo-negative samples.") if DEBUG else None
    print(f"New dataset shape: {augmented_data.shape}") if DEBUG else None

    return augmented_data


def standardize(
    data: pd.DataFrame, target_size: tuple = (60, 80)
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize, standardize, and center the dataset.

    Args:
        data (pd.DataFrame): Dataframe with the last 3 columns of the features.csv file.
        target_size (tuple, optional): Target size (height, width) for resizing images.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - A 4D NumPy array of processed images with shape (num_images, height, width, channels).
            - A 2D NumPy array of corresponding labels.
    """
    features = []  # Use a list to store images

    num_images = len(data)
    for i in range(num_images):
        current_image = data["file_names"].iloc[i]

        # Handle and reshape the TIFF image
        meaned_image = tiff_handling.handle_tiff(current_image)  # Shape: (H, W)
        meaned_image = meaned_image[
            ..., np.newaxis
        ]  # Add a channel dimension, Shape: (H, W, 1)

        # Resize the image to the target size
        resized_image = tf.image.resize(meaned_image, target_size, method="bilinear")
        print("Resulting img shape: ", resized_image.shape) if DEBUG else None

        # Convert to NumPy array and append to the list
        features.append(resized_image.numpy())

    # Convert list to NumPy array, Shape: (num_images, height, width, channels)
    features_array = np.array(features)

    # creating the image data generator to standardize images
    datagen = ImageDataGenerator(
        featurewise_center=True, featurewise_std_normalization=True
    )

    datagen.fit(features_array)

    # Extract labels
    labels = data[["classification"]].values  # Convert DataFrame column to NumPy array
    labels = np.where(labels == 1, 0, 1)

    # Return processed images and labels
    return features_array, labels


def visualize(image: pd.DataFrame, method: str = "standardize"):
    img = tiff.imread(image["file_names"].iloc[0])

    for i, page in enumerate(img):
        plt.imshow(page, cmap="gray")
        plt.title(f"Page {i}")
        plt.show()

    if method == "VGG":
        extracted_features = VGG_features(image)
    else:
        extracted_features, _ = standardize(image)

    plt.figure(figsize=(60, 80))
    plt.title("Extracted features")
    plt.imshow(extracted_features, cmap="gray")
    plt.show()


if __name__ == "__main__":
    sample_data = load_csv.sample_data(
        load_csv.get_correct_data(load_csv.load_pandas()), 5
    )
    trainX, trainy = standardize(sample_data)
    print("trainX shape: ", trainX.shape)
    print("trainy shape: ", trainy.shape)
    print("trainy: ", trainy)

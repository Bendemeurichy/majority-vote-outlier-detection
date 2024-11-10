import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import pandas as pd
import numpy as np

from utils import dataloader

image_path = "./HT29_data/images/"


def filter_single_cells(data: pd.DataFrame) -> pd.DataFrame:
    """Filter the data to only include single cells.

    Args:
        data (pd.DataFrame): Data to filter.

    Returns:
        pd.DataFrame: Filtered data.
    """
    return data[data["classification"] == 1]


# TODO: Make use of csv file to open
def VGG_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Extracts features from the dataset(images) using a pretrained VGG16 model.

    Args:
        dataset (pd.DataFrame): Dataframe with the last 3 columns of the features.csv file.
        batch_size (int, optional): Batch size to process the images(adapt besed on available memory). Defaults to 32.

    Returns:
        pd.DataFrame: Dataframe with the extracted features([]).
    """
    model = VGG16(weights="imagenet", include_top=False)

    target_size = (224, 224)

    all_features = []

    print(f"Extracting features from {len(dataset)} images")

    # Process the images
    num_images = len(dataset)
    for i in range(0, num_images):
        current_image = dataset["file_names"].iloc[i]  # Use iloc for indexing

        print(f"Loading image: {image_path + current_image}")

        # Load and preprocess the image
        img = image.load_img(image_path + current_image, target_size=target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)  # Add batch dimension
        x = preprocess_input(x)  # Preprocess image for VGG16

        # Extract features
        current_features = model.predict(x)

        # Flatten the features into a 1D array
        flattened_features = current_features.flatten()

        # Append to the list of features
        all_features.append(flattened_features)

    # Convert list of features into a DataFrame
    features_df = pd.DataFrame(all_features)

    # Concatenate the labels from the dataset (ensure index alignment)
    output_df = pd.concat(
        [features_df, dataset[["label"]].reset_index(drop=True)], axis=1
    )

    print(f"Extracted features shape: {output_df.shape}")

    # print(output_df)

    return output_df


if __name__ == "__main__":
    sample_data = dataloader.sample_data(dataloader.load_pandas(), 20)
    single_cells = filter_single_cells(sample_data)
    output = VGG_features(single_cells)
    # print(output.apply(np.flatnonzero, axis=1))
    df_non_zero = output.loc[:, (output != 0).any(axis=0)]
    print(df_non_zero)
    # print(output)

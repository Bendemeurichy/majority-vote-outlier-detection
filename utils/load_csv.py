import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch


def get_project_root() -> Path:
    """Find the project root by looking for certain marker files/directories"""
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        # Check for common project markers - adjust these as needed
        if (parent / "HT29_data").exists():  # Using data dir as marker
            return parent
    raise FileNotFoundError("Could not find project root directory")


def load_pandas() -> pd.DataFrame:
    project_root = get_project_root()
    data_path = project_root / "HT29_data" / "features.csv"
    return pd.read_csv(data_path, delimiter=";").iloc[:, -3:]


def split_data(
    data: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Split the data into training, testing, and validation sets.

    Args:
        data (pd.DataFrame): Data to split.
        test_size (float): Proportion of data to be used for the test set.
        val_size (float): Proportion of data to be used for the validation set.
        random_state (int): Seed for reproducibility.

    Returns:
        dict: Dictionary with keys 'train', 'test', and 'val' containing the respective datasets.
    """
    # Split into training + temp (test + validation)
    train_data, temp_data = train_test_split(
        data, test_size=test_size + val_size, random_state=random_state
    )

    # Now split the temp_data into test and validation sets
    test_data, val_data = train_test_split(
        temp_data,
        test_size=val_size / (test_size + val_size),
        random_state=random_state,
    )

    return (train_data, test_data, val_data)


def get_correct_data(data: pd.DataFrame) -> pd.DataFrame:
    """Get the data that has the correct classification.

    Args:
        data (pd.DataFrame): Data to filter.

    Returns:
        pd.DataFrame: Data with the correct classification.
    """
    return data[data["classification"] == 1]


def sample_correct_data(data: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """Sample n_samples from the data that has the correct classification.

    Args:
        data (pd.DataFrame): Data to sample from.
        n_samples (int): Number of samples to take.

    Returns:
        pd.DataFrame: Sampled data with the correct classification.
    """
    return get_correct_data(data).sample(n_samples)


def sample_data(data: pd.DataFrame, n_samples: int, random_state=42) -> pd.DataFrame:
    """Sample n_samples from the data.

    Args:
        data (pd.DataFrame): Data to sample from.
        n_samples (int): Number of samples to take.

    Returns:
        pd.DataFrame: Sampled data.
    """
    return data.sample(n_samples, random_state=random_state)


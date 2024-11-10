import pandas as pd


def load_pandas() -> pd.DataFrame:
    """Load the last 3 columns of the features.csv file.
        Returns a dataframe with the classification, filename and the label.

    Returns:
        pd.DataFrame: Dataframe with the last 3 columns of the features.csv file.
    """

    return pd.read_csv("./HT29_data/features.csv", delimiter=";").iloc[:, -3:]


def sample_data(data: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """Sample n_samples from the given data.

    Args:
        data (pd.DataFrame): Data to sample from.
        n_samples (int): Number of samples to take.

    Returns:
        pd.DataFrame: Sampled data.
    """
    return data.sample(n_samples)

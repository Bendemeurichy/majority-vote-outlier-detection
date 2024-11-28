import pickle
from minisom import MiniSom
import numpy as np
from tqdm import tqdm


class SOM:
    def __init__(self, input_dim, map_dim, sigma=1.0, learning_rate=0.5):
        """
        SOM-based anomaly detector using MiniSom.

        Args:
            input_dim (int): Dimensionality of input data.
            map_dim (int): Size of the square SOM grid (map_dim x map_dim neurons).
            sigma (float): Spread of the neighborhood function (default: 1.0).
            learning_rate (float): Initial learning rate (default: 0.5).
        """
        self.map_dim = map_dim
        self.input_dim = input_dim
        self.som = MiniSom(
            map_dim, map_dim, input_dim, sigma=sigma, learning_rate=learning_rate
        )
        self.threshold = None  # Initialize threshold as None

    def train(self, data, num_epochs=1000):
        """
        Train the SOM model.

        Args:
            data (numpy.ndarray): Input data of shape (num_samples, input_dim).
            num_epochs (int, optional): Number of training epochs. Defaults to 1000.
        """
        print("Training SOM...")
        self.som.random_weights_init(data)
        self.som.train_batch(data, num_epochs, verbose=True)
        print("Training complete.")

    def compute_quantization_error(self, data):
        """
        Compute the quantization error of the SOM for the given dataset.

        Args:
            data (numpy.ndarray): Input data of shape (num_samples, input_dim).

        Returns:
            float: Quantization error.
        """

        errors = np.array(
            [
                np.linalg.norm(data[i] - self.som.get_weights()[self.som.winner(data[i])])
                for i in range(len(data))
            ]
        )
        quantization_error = np.mean(errors)
        return quantization_error

    def compute_bmu_distance(self, data):
        """
        Compute the bmu distances.

        Args:
            data (numpy.ndarray): Input data of shape (num_samples, input_dim).

        Returns:
            numpy.ndarray: Reconstruction errors for each input sample.
        """
        errors = np.array(
            [
                np.linalg.norm(
                    data[i] - self.som.get_weights()[self.som.winner(data[i])]
                )
                for i in range(len(data))
            ]
        )
        return errors

    def set_threshold(self, data, percentile=95):
        """
        Set anomaly threshold based on bmu distances from normal data.

        Args:
            data (numpy.ndarray): Normal training data.
            percentile (int, optional): Percentile for the threshold. Defaults to 95.

        Returns:
            float: Threshold value.
        """
        errors = self.compute_bmu_distance(data)
        self.threshold = np.percentile(errors, percentile)
        print(f"Threshold set at {self.threshold:.4f} (Percentile: {percentile})")
        return self.threshold

    def predict(self, data):
        """
        Predict whether samples are anomalies or not.

        Args:
            data (numpy.ndarray): Input data of shape (num_samples, input_dim).

        Returns:
            numpy.ndarray: Binary predictions (1 = anomaly, 0 = normal).
        """
        errors = self.compute_bmu_distance(data)
        predictions = (errors > self.threshold).astype(int)
        return predictions, errors

    def save_model(self, path):
        """
        Save the trained SOM model to a file.

        Args:
            path (str): File path to save the model.
        """
        with open(path, "wb") as f:
            pickle.dump({"som": self.som, "threshold": self.threshold}, f)
        print(f"Model saved to {path}.")

    def load_model(self, path):
        """
        Load a trained SOM model from a file.

        Args:
            path (str): File path to load the model from.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.som = data["som"]
            self.threshold = data["threshold"]
        print(f"Model loaded from {path}.")

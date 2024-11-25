import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from models.CNN import feature_extraction as fe
from utils import load_csv


class CNN:
    def __init__(self, data: pd.DataFrame = pd.DataFrame([])):
        self.data = data
        self.model = None
        self.test_data = pd.DataFrame([])

    def load_data(
        self,
        data_fraction: float = 1.0,
        negative_data_fraction: float = 1.0,
        target_size: tuple = (60, 80),
    ):
        """This function loads the data from the csv file and preprocesses it, adding pseudo-negative data as well.
        Attention: From the loaded singlet data 10% will always be reserved as a test set in advance.
                    To this data the actual outlier data will be added(since the amount of outliers is approximately 10% of the inliers), this way we avoid testing on peusdo negative data.

        Args:
            data_fraction (float, optional): defines what precentage of the data you would want to load. Defaults to 1.0.
            negative_data_fraction (float, optional): defines what precentage of the pseudo-negative data you would want to add. Defaults to 1.0(which is equivalent to the amount of the positive data).
            target_size (tuple, optional): Target size(height, width) for the images. Defaults to (60, 80).
        """
        # Load all the data and separate inliers and outliers
        all_data = load_csv.load_pandas()
        all_inliers = load_csv.get_correct_data(all_data)
        all_outliers = load_csv.get_outliers(all_data)

        if data_fraction < 1.0:
            inlier_size = int(data_fraction * len(all_inliers))
            outlier_size = int(data_fraction * len(all_outliers))
            data = load_csv.sample_data(all_inliers, inlier_size)
            outliers = load_csv.sample_data(all_outliers, outlier_size)

        else:
            data = all_inliers
            outliers = all_outliers

        inlier_features = fe.VGG_features(data, target_size=target_size)
        outlier_features = fe.VGG_features(outliers, target_size=target_size)

        # Reserve 10% of the total inlier data for testing so that not pseudo-negative data is included
        self.test_data = inlier_features.sample(n=outlier_size, random_state=42)

        inlier_features = inlier_features.drop(self.test_data.index)

        # Append all of the outlier data to the test data(we chose to append all of the
        # (relative) outlier data since the amuont of outliers is appriximately equal to 10% of the inliers)
        self.test_data = pd.concat([self.test_data, outlier_features])

        print(f"The training set size is: {len(inlier_features)}")
        print(f"The amount of outliers in the test set is: {len(outlier_features)}")
        print(
            f"The amount of inliers in the test set is: {len(self.test_data) - len(outlier_features)}"
        )

        self.data = fe.append_pseudo_negative_data(
            inlier_features, fraction=negative_data_fraction
        )

    def build(self):
        """Build the model.

        Raises:
            ValueError: Data has not been loaded yet.
        """

        print("--------------------- Building the model ---------------------")

        if self.data.empty:
            raise ValueError("Data has not been loaded yet.")

        # Determine the input dimension from the feature shape
        input_dim = self.data.drop("classification", axis=1).shape[1]

        print(f"Input dimension: {input_dim}")

        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(input_dim,)),
                tf.keras.layers.Dense(
                    units=input_dim, activation="relu"
                ),  # Fully connected layer
                tf.keras.layers.Dense(
                    units=2, activation="softmax"
                ),  # Output layer for 2 classes
            ]
        )

        self.model = model

    def train(self, train_size=0.7, val_size=0.3, random_state=42):
        """_summary_

        Args:
            train_size (float, optional): fractional size of the training set. Defaults to 0.7.
            val_size (float, optional): fractional size of the validation set. Defaults to 0.15.
            random_state (int, optional): . Defaults to 42.

        Raises:
            ValueError: Train, validation, and test sizes must sum to 1.
            ValueError: Data has not been loaded yet.
            ValueError: Model has not been built yet.
        """
        if not np.isclose(train_size + val_size, 1.0):
            raise ValueError("Train, validation, and test sizes must sum to 1.")
        if self.data.empty:
            raise ValueError("Data has not been loaded yet.")
        if self.model is None:
            raise ValueError("Model has not been built yet.")

        # Split the data into training and validation
        train_data, val_data = train_test_split(
            self.data, test_size=val_size, random_state=random_state
        )

        # Split the training data into features and labels
        X_train = train_data.drop("classification", axis=1)
        y_train = train_data["classification"]

        # Split the validation data into features and labels
        X_val = val_data.drop("classification", axis=1)
        y_val = val_data["classification"]

        # Split the test data into features and labels
        X_test = self.test_data.drop("classification", axis=1)
        y_test = self.test_data["classification"]

        # Map labels for binary classification (0: inliers, 1: outliers)
        y_train = y_train.apply(lambda y: 0 if y == 1 else 1)
        y_val = y_val.apply(lambda y: 0 if y == 1 else 1)
        y_test = y_test.apply(lambda y: 0 if y == 1 else 1)

        # Compile the model
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Train the model
        self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=32,
            verbose=2,
        )

        # Evaluate the model
        self.evaluate(X_test, y_test)

    def evaluate(self, X_test, y_test):
        """Evaluates the model on the test data.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test labels.
        """
        if self.model is None:
            raise ValueError("Model has not been built yet.")

        # Evaluate the model on the test data
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=2)
        print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    model = CNN()
    model.load_data(data_fraction=0.1)
    model.build()
    print("model built")
    model.train()
    print("model trained")

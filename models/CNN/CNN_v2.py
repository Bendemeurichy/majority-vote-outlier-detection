import pandas as pd
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from models.CNN import feature_extraction as fe
from utils import load_csv


class CNN_v2:
    def __init__(self, data: pd.DataFrame = pd.DataFrame([])):
        self.data = data
        self.model = None

        self.X_train: np.ndarray
        self.X_val: np.ndarray
        self.X_test: np.ndarray
        self.y_train: np.ndarray
        self.y_val: np.ndarray
        self.y_test: np.ndarray

    def load_data(
        self,
        data_fraction: float = 1.0,
        target_size: tuple = (60, 80),
        val_size: float = 0.2,
        test_size: float = 0.0,
        test_set: pd.DataFrame = pd.DataFrame([]),
        use_smote: bool = True,
    ):
        # Load all the data and separate inliers and outliers
        all_data = load_csv.load_pandas()
        all_inliers = load_csv.get_correct_data(all_data)
        all_outliers = load_csv.get_outliers(all_data)

        if data_fraction < 1.0:
            inlier_size = int(data_fraction * len(all_inliers))
            data = load_csv.sample_data(all_inliers, inlier_size)
            if inlier_size > len(all_outliers):
                outliers = all_outliers
            else:
                outliers = load_csv.sample_data(all_outliers, inlier_size)

        else:
            data = all_inliers
            outliers = all_outliers

        total_data = pd.concat([data, outliers], ignore_index=True)

        (features_X, features_y) = fe.standardize(total_data, target_size=target_size)

        if use_smote:
            sm = SMOTE(random_state=42, sampling_strategy="minority")

            flattened_features_X = features_X.reshape(
                (features_X.shape[0], features_X.shape[1] * features_X.shape[2])
            )

            smote_x, smote_y = sm.fit_resample(flattened_features_X, features_y)

            features_X = smote_x.reshape(
                smote_x.shape[0], target_size[0], target_size[1], 1
            )
            features_y = smote_y

        if test_set.size > 0:
            # Standardize the test set
            test_features_X, test_features_y = fe.standardize(
                test_set, target_size=target_size
            )
            X_test = test_features_X
            y_test = test_features_y.reshape(-1)

            # Flatten test features for comparison
            flattened_test_features_X = X_test.reshape(
                (X_test.shape[0], X_test.shape[1] * X_test.shape[2])
            )

            # Find indices of test data in the SMOTE-generated data
            smote_flattened_features_X = features_X.reshape(
                (features_X.shape[0], features_X.shape[1] * features_X.shape[2])
            )
            mask = ~np.isin(smote_flattened_features_X, flattened_test_features_X).all(
                axis=1
            )

            # Filter out test features and labels from the SMOTE data
            features_X = features_X[mask]
            features_y = features_y[mask]
        else:
            X_test = np.array([])
            y_test = np.array([])

        X_train, X_val, y_train, y_val = train_test_split(
            features_X, features_y, test_size=val_size, random_state=42
        )

        # Saving the split data
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Validation labels shape: {y_val.shape}")
        print(f"Test labels shape: {y_test.shape}")

    def train(
        self,
        epochs: int = 10,
    ):
        """Train the model

        Args:
            epochs (int, optional): amount of epochs used to train the model. Defaults to 10.

        Raises:
            ValueError: Data has not been loaded yet
        """
        if self.X_train.size == 0:
            raise ValueError("Data has not been loaded yet")

        datagen = ImageDataGenerator(
            featurewise_center=True, featurewise_std_normalization=True
        )

        datagen.fit(self.X_train)

        train_iterator = datagen.flow(self.X_train, self.y_train, batch_size=64)
        val_iterator = datagen.flow(self.X_val, self.y_val, batch_size=64)

        width, height, channels = self.X_train.shape[1], self.X_train.shape[2], 1

        model = Sequential()
        model.add(
            Conv2D(32, (3, 3), activation="relu", input_shape=(width, height, channels))
        )
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(
            Dense(1, activation="sigmoid")
        )  # Output layer for binary classification

        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        model.fit(
            train_iterator,
            steps_per_epoch=len(train_iterator),
            validation_data=val_iterator,
            validation_steps=len(val_iterator),
            epochs=epochs,
        )

        self.model = model

    def predict(self, data: np.ndarray):
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        return self.model.predict(data)

    def evaluate(self) -> tuple[np.ndarray, float]:
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        test_iterator = ImageDataGenerator().flow(
            self.X_test, self.y_test, batch_size=64
        )
        _, acc = self.model.evaluate(test_iterator, steps=len(test_iterator), verbose=0)

        y_predict = self.model.predict(self.X_test, batch_size=64)

        # Print test accuracy
        print("Test Accuracy: %.3f" % (acc * 100))

        # Return predictions and accuracy
        return y_predict, acc

    def save_model(self, path: str):
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        self.model.save(path)

    def load_model(self, path: str):
        self.model = tf.keras.models.load_model(path)


if __name__ == "__main__":
    model = CNN_v2()
    all_data = load_csv.load_pandas()
    all_outliers = load_csv.get_outliers(all_data)
    test_set = load_csv.sample_data(all_outliers, 100)
    model.load_data(test_set=test_set)
    model.train()

    model.evaluate()

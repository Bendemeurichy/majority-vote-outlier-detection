import joblib
import torch

class BaseModel:
    def __init__(self, model=None):
        """
        Initialize the base model with any model type.
        :param model: A machine learning model instance (e.g., scikit-learn, PyTorch, etc.).
        """
        self.model = model

    def fit(self, X, y):
        """
        Fit the model to the training data, if supported.
        :param X: Training data features.
        :param y: Training data labels.
        :return: self
        """
        if hasattr(self.model, 'fit'):
            self.model.fit(X, y)
        else:
            raise NotImplementedError("The model does not support fitting.")
        return self

    def predict(self, X):
        """
        Predict using the trained model, if supported.
        :param X: Input data for prediction.
        :return: Predicted values.
        """
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        elif hasattr(self.model, '__call__'):  # For PyTorch or similar models
            with torch.no_grad():
                return self.model(X)
        else:
            raise NotImplementedError("The model does not support prediction.")

    def score(self, X, y):
        """
        Calculate the model's score, if supported.
        :param X: Test data features.
        :param y: True labels for scoring.
        :return: Model score.
        """
        if hasattr(self.model, 'score'):
            return self.model.score(X, y)
        else:
            raise NotImplementedError("Scoring is not supported by this model.")

    def save(self, path):
        """
        Save the model to a file.
        :param path: Path where the model will be saved.
        """
        if isinstance(self.model, torch.nn.Module):
            torch.save(self.model.state_dict(), path)
        else:
            joblib.dump(self.model, path)

    def load(self, path, model_type=None):
        """
        Load the model from a file.
        :param path: Path to load the model from.
        :param model_type: Type of model to load (for frameworks like PyTorch).
        :return: self
        """
        if model_type == 'torch' and isinstance(self.model, torch.nn.Module):
            self.model.load_state_dict(torch.load(path))
        else:
            self.model = joblib.load(path)
        return self

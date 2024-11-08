### File that contains basic functions to load and save pytorch models

import torch


def save_model(model, path):
    """Save the model to the given path.

    Args:
        model (torch.nn.Module): Model to save.
        path (str): Path to save the model.
    """
    torch.save(model.state_dict(), path)


def load_model(model, path, eval: bool = True) -> torch.nn.Module:
    """Load the model from the given path. If eval is True, the model will be set to evaluation mode.

    Args:
        model (torch.nn.Module): Model to load.
        path (str): Path to load the model from.
        eval (bool, optional): Whether to set the model to evaluation mode or not. Defaults to True.
    """
    model.load_state_dict(torch.load(path))

    if eval:
        model.eval()

    return model

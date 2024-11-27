import optuna
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def train_and_evaluate_som(data, x_dim, y_dim, sigma, learning_rate, iterations):
    """
    Train and evaluate a SOM using Quantization Error (QE) and Topographic Error (TE).
    """
    som = MiniSom(
        x=x_dim,
        y=y_dim,
        input_len=data.shape[1],
        sigma=sigma,
        learning_rate=learning_rate,
    )
    som.random_weights_init(data)
    som.train_random(data, iterations)

    # Compute Quantization Error (QE)
    qe = np.mean([som.distance_from_weights(sample) for sample in data])

    # Compute Topographic Error (TE)
    def topographic_error(data):
        errors = 0
        for sample in data:
            bmus = som.winner(sample, return_neighbors=True)
            if len(bmus) > 1 and np.linalg.norm(bmus[0] - bmus[1]) > 1:
                errors += 1
        return errors / len(data)

    te = topographic_error(data)

    # Combine QE and TE (e.g., weighted sum)
    score = qe + 0.5 * te  # Adjust weight as needed
    return score


def objective(trial):
    """
    Optuna objective function to optimize SOM hyperparameters.
    """
    # Define the search space
    x_dim = trial.suggest_int("x_dim", 5, 50)  # SOM grid x-dimension
    y_dim = trial.suggest_int("y_dim", 5, 50)  # SOM grid y-dimension
    sigma = trial.suggest_float("sigma", 0.1, 5.0)  # Neighborhood radius
    learning_rate = trial.suggest_float(
        "learning_rate", 0.01, 0.5
    )  # Initial learning rate
    iterations = trial.suggest_int(
        "iterations", 1000, 10000, step=500
    )  # Training iterations

    # Train and evaluate the SOM
    score = train_and_evaluate_som(data, x_dim, y_dim, sigma, learning_rate, iterations)
    return score


# Load and preprocess data
# TODO: Load data when som training is implemented and works on hpc

study = optuna.create_study(
    direction="minimize",
    study_name="VAE",
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30),
)
study.optimize(objective, n_trials=1000, timeout=600)


print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")


import optuna.visualization as vis

vis.plot_optimization_history(study).write_image("optimization_history.png")
vis.plot_param_importances(study).write_image("param_importances.png")
vis.plot_intermediate_values(study).write_image("intermediate_values.png")
vis.plot_timeline(study).write_image("timeline.png")
vis.plot_rank(study).write_image("rank.png")

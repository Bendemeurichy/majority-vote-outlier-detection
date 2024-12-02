import torch
import optuna

from torchvision.transforms import v2

import sys
sys.path.append('../../../')

import utils.load_csv as load_csv
from utils.dataloader import ImagePathDataset
from models.VAE.VAE import VAE



set = load_csv.load_pandas()

train,val,test = load_csv.split_data(set)
print(f"train length: {len(train)}")
print(f"val length: {len(val)}")
print(f"test length: {len(test)}")

print(f'outlier test training: {any(train["classification"] != 1)}')
print(f'outlier test validation: {any(val["classification"] != 1)}')
print(f'outlier test test: {any(test["classification"] != 1)}')

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 4800
    training_size = len(train)
    batch_size = trial.suggest_categorical("batch_size", [training_size//10, training_size//5, training_size//2])
    hidden_dim = trial.suggest_int("hidden_dim", 200, 4000, step=200)
    latent_dim = trial.suggest_int("latent_dim", 10, 400, step=10)
    epochs = trial.suggest_int("epochs", 50, 500,step=50)

    learning_rate = 1e-4
    weight_decay = 1e-2

    model = VAE(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((60, 80)),
            v2.ToTensor(),
            v2.Lambda(
                lambda x: (x.view(-1) - torch.min(x)) / (torch.max(x) - torch.min(x))
            ),
        ]
    )

    train_set = ImagePathDataset(train, transform=transform)
    val_set = ImagePathDataset(val, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True
    )

    prev_updates = 0
    best_val_loss = float("inf")
    patience, early_stop_counter = 5, 0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        prev_updates = model.train_model(train_dataloader, optimizer, prev_updates)
        val_loss = model.test(val_dataloader, prev_updates)
        best_val_loss = min(val_loss, best_val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping...")
                break

        # Optional pruning
        trial.report(best_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss

study = optuna.create_study(direction="minimize", 
                            study_name="VAE",
                            sampler=optuna.samplers.TPESampler(),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30))
study.optimize(objective, n_trials=1000, timeout=6000,)

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
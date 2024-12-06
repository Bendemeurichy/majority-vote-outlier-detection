import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
from skimage.metrics import structural_similarity as ssim

# code from: https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/


class VAE(nn.Module):
    """Variational AutoEncoder model using PyTorch.
        The mdodel will be used to detect anomalies in the dataset.
        This means it will be trained on normal data and then used to predict if the input data is an anomaly or not.

    Args:
        input_dim (int): Number of input dimensions.
        hidden_dim (int): Number of hidden dimensions.
        latent_dim (int): Number of latent dimensions.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, treshold=None):
        super(VAE, self).__init__()

        self.treshold = treshold

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 8, 2 * latent_dim),  # 2 for mean and variance.
        )
        self.softplus = nn.Softplus()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 8),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 8, hidden_dim // 4),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x, eps: float = 1e-8) -> torch.distributions.MultivariateNormal:
        """Encode the input data.

        Args:
            x (torch.Tensor): Input data.
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
        Returns:
            torch.distributions.MultivariateNormal: Normal distribution of the encoded data.
        """
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=-1)
        std = self.softplus(log_var) + eps
        scale_tril = torch.diag_embed(std)

        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)

    def reparameterize(self, dist):
        """
        Reparameterizes the encoded data to sample from the latent space.

        Args:
            dist (torch.distributions.MultivariateNormal): Normal distribution of the encoded data.
        Returns:
            torch.Tensor: Sampled data from the latent space.
        """
        return dist.rsample()

    def decode(self, z):
        """Decode the the data from the latent space.

        Args:
            z (torch.Tensor): Sampled data from the latent space.
        Returns:
            torch.Tensor: Decoded data in the original space.
        """

        return self.decoder(z)

    def forward(self, x, compute_loss: bool = True):
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input data.
            compute_loss (bool, optional): Wheter to compute the loss or not. Defaults to True.
        """
        distribution = self.encode(x)
        sample = self.reparameterize(distribution)
        x_reconstructed = self.decode(sample)

        if not compute_loss:
            return {
                "distribution": distribution,
                "sample": sample,
                "x_reconstructed": x_reconstructed,
                "loss": None,
                "loss_reconstruction": None,
                "loss_kl": None,
            }

        loss_reconstruction = (
            F.binary_cross_entropy(x_reconstructed, x, reduction="none")
            .sum(dim=-1)
            .mean()
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(sample).to(device), torch.eye(sample.shape[-1]).to(device)
        )

        loss_kl = torch.distributions.kl.kl_divergence(distribution, std_normal).mean()
        loss = loss_reconstruction + loss_kl

        return {
            "distribution": distribution,
            "sample": sample,
            "x_reconstructed": x_reconstructed,
            "loss": loss,
            "loss_reconstruction": loss_reconstruction,
            "loss_kl": loss_kl,
        }

    def train_model(self, dataloader, optimizer, prev_updates, writer=None):
        """
        Trains the model on the given data.

        Args:
            model (nn.Module): The model to train.
            dataloader (torch.utils.data.DataLoader): The data loader.
            loss_fn: The loss function.
            optimizer: The optimizer.
        """
        self.train()  # Set the model to training mode
        device = next(self.parameters()).device

        for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
            n_upd = prev_updates + batch_idx

            data = data.to(device)

            optimizer.zero_grad()  # Zero the gradients

            output = self(data)  # Forward pass
            loss = output["loss"]  # Compute the loss

            loss.backward()

            if n_upd % 100 == 0:
                # Calculate and log gradient norms
                total_norm = 0.0
                for p in self.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1.0 / 2)

                print(
                    f"Step {n_upd:,} (N samples: {n_upd*dataloader.batch_size:,}), Loss: {loss.item():.4f} (Recon: {output['loss_reconstruction'].item():.4f}, KL: {output['loss_kl'].item():.4f}) Grad: {total_norm:.4f}"
                )

                if writer is not None:
                    global_step = n_upd
                    writer.add_scalar("Loss/Train", loss.item(), global_step)
                    writer.add_scalar(
                        "Loss/Train/BCE",
                        output["loss_reconstruction"].item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "Loss/Train/KLD", output["loss_kl"].item(), global_step
                    )
                    writer.add_scalar("GradNorm/Train", total_norm, global_step)

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

            optimizer.step()  # Update the model parameters

        return prev_updates + len(dataloader)

    def test(self, dataloader, cur_step, writer=None):
        """
        Tests the model on the given data.

        Args:
            dataloader (torch.utils.data.DataLoader): The data loader.
            cur_step (int): The current step.
            writer: The TensorBoard writer.
        """
        self.eval()  # Set the model to evaluation mode
        device = next(self.parameters()).device
        test_loss = 0
        test_recon_loss = 0
        test_kl_loss = 0

        with torch.no_grad():
            for data, target in tqdm(dataloader, desc="Testing"):
                data = data.to(device)
                data = data.view(data.size(0), -1)  # Flatten the data

                output = self(data, compute_loss=True)  # Forward pass

                test_loss += output["loss"].item()
                test_recon_loss += output["loss_reconstruction"].item()
                test_kl_loss += output["loss_kl"].item()

        test_loss /= len(dataloader)
        test_recon_loss /= len(dataloader)
        test_kl_loss /= len(dataloader)
        print(
            f"====> Test set loss: {test_loss:.4f} (BCE: {test_recon_loss:.4f}, KLD: {test_kl_loss:.4f})"
        )

        if writer is not None:
            writer.add_scalar("Loss/Test", test_loss, global_step=cur_step)
            writer.add_scalar(
                "Loss/Test/BCE",
                output["loss_reconstruction"].item(),
                global_step=cur_step,
            )
            writer.add_scalar(
                "Loss/Test/KLD", output["loss_kl"].item(), global_step=cur_step
            )

            # Log reconstructions
            writer.add_images(
                "Test/Reconstructions",
                output["x_reconstructed"].view(-1, 1, 60, 80),
                global_step=cur_step,
            )
            writer.add_images(
                "Test/Originals", data.view(-1, 1, 60, 80), global_step=cur_step
            )

            # Log random samples from the latent space
            z = torch.randn(16, self.latent_dim).to(device)
            samples = self.decode(z)
            writer.add_images(
                "Test/Samples", samples.view(-1, 1, 60, 80), global_step=cur_step
            )
        return test_loss

    def extract_error_threshold(self, dataloader, method="percentile", value=95):
        """Determine anomaly threshold based on reconstruction errors.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader containing normal data.
            method (str): Method to determine threshold ("percentile", "zscore", "gmm").
            value (float, optional): Percentile or z-score value, or GMM component probability.
                Defaults to 95 for percentile or 3 for z-score.

        Returns:
            float: Threshold value for anomaly detection.
        """
        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        reconstruction_errors = []

        with torch.no_grad():
            for data, _ in tqdm(dataloader, desc="Computing threshold"):
                data = data.to(device)

                # Get model outputs
                output = self(data, compute_loss=False)
                x_reconstructed = output["x_reconstructed"]

                ssim_errors = batch_ssim(data, x_reconstructed)

                # Use structural similarity index to compute reconstruction errors
                reconstruction_errors.extend(ssim_errors)

        reconstruction_errors = np.array(reconstruction_errors)

        if method == "percentile":
            threshold = np.percentile(reconstruction_errors, value)
        elif method == "zscore":
            threshold = np.mean(reconstruction_errors) + value * np.std(
                reconstruction_errors
            )
        elif method == "gmm":
            from sklearn.mixture import GaussianMixture

            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(reconstruction_errors.reshape(-1, 1))
            cluster_means = gmm.means_.flatten()
            threshold = cluster_means[np.argmax(cluster_means)]
        else:
            raise ValueError(f"Unknown method: {method}")

        self.threshold = threshold
        return threshold

    def predict(self, data: pd.DataFrame, threshold=None):
        """Predict if the data contains anomalies.

        Args:
            data (pd.DataFrame): Input data with an 'image' column containing tensors.
            threshold (float, optional): Anomaly threshold. If None, uses the pre-computed threshold.

        Returns:
            pd.DataFrame: Updated with:
                - 'prediction': Anomaly classification (1 = anomaly, 0 = normal).
                - 'reconstruction_error': Reconstruction errors for each sample.
        """
        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = torch.stack(list(data["image"])).to(device)

        with torch.no_grad():
            # Forward pass through the model
            output = self(images, compute_loss=False)
            x_reconstructed = output["x_reconstructed"]

            # Compute reconstruction error using structural similarity index
            reconstruction_errors = batch_ssim(images, x_reconstructed)

            # Classify anomalies based on threshold
            if threshold is None:
                threshold = self.threshold
            predictions = reconstruction_errors < threshold

            # Update DataFrame
            data["prediction"] = predictions.astype(int)

            data["reconstruction_error"] = reconstruction_errors

            return data

        return data


def batch_ssim(originals, reconstructions):
    errors = [
        1
        - ssim(
            originals[i].cpu().numpy().squeeze(),
            reconstructions[i].cpu().numpy().squeeze(),
            data_range=1,
        )
        for i in range(originals.shape[0])
    ]
    return np.array(errors)

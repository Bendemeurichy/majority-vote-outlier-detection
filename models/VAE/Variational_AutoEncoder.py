import torch
import torch.nn as nn
import torch.nn.functional as F

# code from: https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/


class VariationalAutoEncoder(nn.Module):
    """Variational AutoEncoder model using PyTorch.
        The mdodel will be used to detect anomalies in the dataset.
        This means it will be trained on normal data and then used to predict if the input data is an anomaly or not.

    Args:
        input_dim (int): Number of input dimensions.
        hidden_dim (int): Number of hidden dimensions.
        latent_dim (int): Number of latent dimensions.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VariationalAutoEncoder, self).__init__()

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

        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(sample), torch.eye(sample.shape[-1])
        )

        loss_kl = (
            torch.distributions.kl_divergence(distribution, std_normal)
            .sum(dim=-1)
            .mean()
        )
        loss = loss_reconstruction + loss_kl

        return {
            "distribution": distribution,
            "sample": sample,
            "x_reconstructed": x_reconstructed,
            "loss": loss,
            "loss_reconstruction": loss_reconstruction,
            "loss_kl": loss_kl,
        }

    # TODO: Implement the following functions
    def train_normal_data(self, dataloader, optimizer, loss_fn, epochs):
        """Train the model on normal data.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
            optimizer (torch.optim.Optimizer): Optimizer for the model.
            loss_fn (torch.nn.Module): Loss function for the model.
            epochs (int): Number of epochs to train the model.
        """
        pass

    def test(model, dataloader):
        """Test the model on normal data.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for the test data.
        """
        pass

    def extract_error_treshold(self, dataloader):
        """Evaluate the model on normal data.
            Measure the reconstruction error for the normal data.
            Used to set the threshold for the anomaly detection.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation data.
        Returns:
            torch.Tensor: Reconstruction error for the normal data.
        """
        pass

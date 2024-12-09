import sys
from torchvision.transforms import v2
import pandas as pd
import torch
import utils.dataloader as dataloader
import utils.load_csv as load_csv
import numpy as np
import os

sys.path.append("../..")

from SOM.SOM import SOM
from VAE.VAE import VAE
from CNN.CNN_v2 import CNN_v2 as CNN

from utils.load_save_models import load_model
from preprocessing import tiff_handling


class Majority_Vote:
    som_input_dim = 100
    som_map_dim = 80

    som_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((60, 80)),
            # v2.Lambda(
            #     lambda x: (x.view(-1) - torch.min(x)) / (torch.max(x) - torch.min(x))
            # ),
        ]
    )

    vae_batch_size = 436
    vae_input_dim = 4800
    vae_hidden_dim = 4000
    vae_latent_dim = 320

    vae_transform = v2.Compose(
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

    def __init__(
        self,
        som_model_path: str = "models/SOM/hpc_SOM_model.pkl",
        vae_model_path: str = "models/VAE/vae.pth",
        cnn_model_path: str = "models/CNN/cnn.keras",
    ):
        # SOM

        self.som = SOM.load_model(
            SOM(self.som_input_dim, self.som_map_dim), som_model_path
        )

        # VAE

        vae = VAE(self.vae_input_dim, self.vae_hidden_dim, self.vae_latent_dim)

        self.vae = load_model(vae, vae_model_path, True)

        # CNN

        self.cnn = CNN()
        self.cnn.load_model(cnn_model_path)

    def set_thresholds(self, val_set: pd.DataFrame):
        vae_val_set = dataloader.ImagePathDataset(val_set, self.vae_transform)

        som_val_set = np.array(
            [
                tiff_handling.flatten_image(
                    self.som_transform(tiff_handling.handle_tiff(el))
                )
                for el in val_set["file_names"].tolist()
            ]
        )

        val_dataloader = torch.utils.data.DataLoader(
            vae_val_set, batch_size=self.vae_batch_size, shuffle=True
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae.to(device)

        self.vae.extract_error_threshold(val_dataloader, method="percentile", value=85)

        self.som.set_threshold(som_val_set, percentile=90)

    def predict(self, test_set: pd.DataFrame):
        som_test = test_set
        som_test["image"] = som_test["file_names"].apply(
            lambda x: tiff_handling.flatten_image(
                self.som_transform(tiff_handling.handle_tiff(x))
            )
        )

        vae_test = test_set
        vae_test["image"] = vae_test["file_names"].apply(
            lambda x: self.vae_transform(tiff_handling.handle_tiff(x))
        )

        cnn_test = test_set
        self.cnn.load_data(test_set=cnn_test)

        som_output = self.som.predict(som_test)
        vae_output = self.vae.predict(vae_test)
        cnn_output, _ = self.cnn.evaluate()

        som_output = np.array(som_output["prediction"])
        vae_output = np.array(vae_output["prediction"])

        print("Shape of SOM output:", som_output.shape)
        print("Shape of VAE output:", vae_output.shape)
        print("Shape of CNN output:", cnn_output.shape)

        print("Test set shape:", test_set.shape)

        # Majority vote
        majority_vote = np.zeros(len(test_set))

        temp_arr = np.add(som_output, vae_output)
        temp_arr = np.add(temp_arr, cnn_output)
        majority_vote = np.where(temp_arr < 2, 0, 1)

        return majority_vote


if __name__ == "__main__":
    mv = Majority_Vote()
    all_data = load_csv.load_pandas()
    _, val_set, test_set = load_csv.split_data(all_data)
    mv.set_thresholds(val_set)
    predictions = mv.predict(test_set)
    print(predictions)
    print("done")

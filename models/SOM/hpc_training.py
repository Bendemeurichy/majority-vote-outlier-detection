import numpy as np
from torchvision.transforms import v2
import torch
import sys
from SOM import SOM

sys.path.append("../../")

import utils.load_csv as load_csv
import preprocessing.tiff_handling as tiff_handling


def main():
    # Load and preprocess data
    dimensions = 100

    set = load_csv.load_pandas()
    train, val, test = load_csv.split_data(set)

    print(f"train length: {len(train)}")
    print(f"val length: {len(val)}")
    print(f"test length: {len(test)}")

    print(f'outlier test training: {any(train["classification"] != 1)}')
    print(f'outlier test validation: {any(val["classification"] != 1)}')
    print(f'outlier test test: {any(test["classification"] != 1)}')

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((60, 80)),
            # v2.Lambda(
            #     lambda x: (x.view(-1) - torch.min(x)) / (torch.max(x) - torch.min(x))
            # ),
        ]
    )

    train_data = np.array(
        [
            tiff_handling.flatten_image(transform(tiff_handling.handle_tiff(el)))
            for el in train["file_names"].tolist()
        ]
    )

    # TODO use validation set to determine the threshold for the outlier detection

    som = SOM(train_data.shape[1],dimensions)

    som.train(train_data)

    val_data = np.array(
        [
            tiff_handling.flatten_image(transform(tiff_handling.handle_tiff(el)))
            for el in val["file_names"].tolist()
        ]
    )

    quant_error = som.compute_quantization_error(val_data)

    print(f"Quantization error: {quant_error}")

    som.save_model("./hpc_SOM_model.pkl")


if __name__ == "__main__":
    main()

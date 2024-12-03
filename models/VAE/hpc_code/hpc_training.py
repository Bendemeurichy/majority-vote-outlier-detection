import argparse
import sys
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2

sys.path.append("../../../")

from utils.dataloader import ImagePathDataset
import utils.load_csv as load_csv
from models.VAE.VAE import VAE


def main():
    batch_size = 436
    lr = 1e-4
    weight_decay = 1e-2
    epochs = 50
    input_dim = 4800
    hidden_dim = 4000
    latent_dim = 320

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=0)

    args = parser.parse_args()

    set = load_csv.load_pandas()
    if args.samples > 0:
        set = set.sample(n=args.samples)
        

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    writer = SummaryWriter(
        f'runs/outlier/vae_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    )

    prev_updates = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        prev_updates = model.train_model(
            train_dataloader, optimizer, prev_updates, writer
        )
        model.test(val_dataloader, prev_updates, writer)
    
    torch.save(model.state_dict(), "vae.pth")

if __name__ == "__main__":
    main()
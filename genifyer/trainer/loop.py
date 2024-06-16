import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from genifyer.model.VariationalAutoEncoder import VAE


def loss_function(recon_x, x, mu, logvar):
    mse = nn.functional.mse_loss(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld


def train(
    epochs: int, train_dataloader: DataLoader, valid_dataloader: DataLoader,
    model: VAE, optimizer, batch_size: int, device: str
) -> tuple[VAE, list]:
    dataloader_dict = {"Train": train_dataloader, "Valid": valid_dataloader}

    training_data = []

    with tqdm(range(epochs)) as pbar_epoch:
        for epoch in pbar_epoch:
            pbar_epoch.set_description(f"epoch : {epoch + 1}")

            metas = []

            for phase in ["Train", "Valid"]:
                if phase == "Train":
                    model.train()
                else:
                    model.eval()

                epoch_loss = 0.0

                for batch in dataloader_dict[phase]:
                    optimizer.zero_grad()
                    x = batch[0].to(device)

                    with torch.set_grad_enabled(phase == "Train"):
                        recon_x, mu, logvar = model(x)
                        loss = loss_function(recon_x, x, mu, logvar)

                        if phase == "Train":
                            loss.backward()
                            optimizer.step()

                        epoch_loss += loss.item()

                epoch_loss /= len(dataloader_dict[phase].dataset) * batch_size

                meta = {"Loss": epoch_loss}
                metas.append(meta)

            training_data.append(dict(zip(["Train", "Valid"], metas)))

    return model, training_data

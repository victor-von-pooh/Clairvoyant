import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

from genifyer.model.VariationalAutoEncoder import VAE


def plot_data(training_data, out_dir):
    train_y = [
        training_data[i]["Train"]["Loss"] for i in range(len(training_data))
    ]
    valid_y = [
        training_data[i]["Valid"]["Loss"] for i in range(len(training_data))
    ]

    x = [i + 1 for i in range(len(train_y))]

    plt.figure(figsize=(18, 12))
    plt.title("Loss comparison", size=15, color="red")
    plt.grid()

    plt.plot(x, train_y, label="Train")
    plt.plot(x, valid_y, label="Valid")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.)
    plt.savefig(f"{out_dir}/loss_curve.png")


def predict(
    samples: int, cfg: dict, model: VAE, ss: StandardScaler, df: pd.DataFrame
) -> pd.DataFrame:
    with torch.no_grad():
        z = torch.randn(samples, cfg["params"]["latent_dim"])
        generated_data = model.decode(z).numpy()

    generated_data = ss.inverse_transform(generated_data)
    cols = df.columns
    gen_df = pd.DataFrame(generated_data, columns=cols)

    return gen_df

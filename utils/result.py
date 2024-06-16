import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import rel_entr
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
    samples: int, cfg: dict, model: VAE, ss: StandardScaler, cols: list
) -> pd.DataFrame:
    with torch.no_grad():
        z = torch.randn(samples, cfg["params"]["latent_dim"])
        generated_data = model.decode(z).numpy()

    generated_data = ss.inverse_transform(generated_data)
    gen_df = pd.DataFrame(generated_data, columns=cols)

    return gen_df


def kl_divergence_evaluation(
    bins: int, cols: list, org_df: pd.DataFrame, pred_df: pd.DataFrame
) -> float:
    org_distributions = []
    pred_distributions = []

    for feat in cols:
        org_hist, _ = np.histogram(org_df[feat], bins=bins, density=True)
        pred_hist, _ = np.histogram(pred_df[feat], bins=bins, density=True)

        org_distributions.append(org_hist / org_hist.sum())
        pred_distributions.append(pred_hist / pred_hist.sum())

    org_distributions = np.array(org_distributions)
    pred_distributions = np.array(pred_distributions)

    kl_divergences = []

    for org, pred in zip(org_distributions, pred_distributions):
        kl_div = np.sum(rel_entr(org, pred))
        kl_divergences.append(kl_div)

    mean_kl_divergence = np.mean(kl_divergences)

    return float(mean_kl_divergence)

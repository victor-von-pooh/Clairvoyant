from typing import Literal

import pandas as pd
import torch
from torch.utils.data import TensorDataset


def df_index_resetter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index().drop("index", axis=1)
    return df


def to_torch(
    df: pd.DataFrame, torch_type: Literal["Float", "Long"]="Float"
) -> torch.Tensor:
    data = df.to_numpy()

    if torch_type=="Float":
        data = torch.from_numpy(data).float()
    else:
        data = torch.from_numpy(data).long()
    
    return data


def make_datasets(cfg: dict) -> dict:
    df = pd.read_csv(cfg["data_path"])
    pre_df = df.copy()

    sampling_rate = cfg["sampling_rate"]["train"] + cfg["sampling_rate"]["valid"]
    train_rate = cfg["sampling_rate"]["train"] / sampling_rate
    sampled_df = pre_df.sample(frac=sampling_rate, random_state=cfg["seed"])

    train_df = sampled_df.sample(frac=train_rate, random_state=cfg["seed"])
    train_index = train_df.index

    valid_df = sampled_df.drop(train_index)

    train_df = df_index_resetter(train_df)
    valid_df = df_index_resetter(valid_df)

    org_torch = to_torch(df, cfg["torch_type"])
    train_torch = to_torch(train_df, cfg["torch_type"])
    valid_torch = to_torch(valid_df, cfg["torch_type"])

    org_dataset = TensorDataset(org_torch)
    train_dataset = TensorDataset(train_torch)
    valid_dataset = TensorDataset(valid_torch)

    out = {
        "org_data": df,
        "org_dataset": org_dataset,
        "train_data": train_df,
        "train_dataset": train_dataset,
        "valid_data": valid_df,
        "valid_dataset": valid_dataset
    }

    return out

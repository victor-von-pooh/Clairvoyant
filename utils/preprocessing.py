from typing import Literal

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset


def df_index_resetter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index().drop("index", axis=1)
    return df


def to_torch(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    torch_type: Literal["Float", "Long"] = "Float",
    scale: bool = False,
) -> tuple[torch.Tensor, StandardScaler]:
    data = df.to_numpy()

    if scale:
        ss = StandardScaler()
        ss.fit(data)
        data = ss.transform(data)
    else:
        ss = scaler
        data = ss.transform(data)

    if torch_type == "Float":
        data = torch.from_numpy(data).float()
    else:
        data = torch.from_numpy(data).long()

    return data, ss


def make_datasets(cfg: dict) -> dict:
    df = pd.read_csv(cfg["data_path"])
    pre_df = df.copy()

    sampling_rate = (
        cfg["sampling_rate"]["train"] + cfg["sampling_rate"]["valid"]
    )
    train_rate = cfg["sampling_rate"]["train"] / sampling_rate
    sampled_df = pre_df.sample(frac=sampling_rate, random_state=cfg["seed"])

    train_df = sampled_df.sample(frac=train_rate, random_state=cfg["seed"])
    train_index = train_df.index

    valid_df = sampled_df.drop(train_index)

    train_df = df_index_resetter(train_df)
    valid_df = df_index_resetter(valid_df)

    train_torch, ss = to_torch(
        train_df, torch_type=cfg["torch_type"], scale=True
    )
    valid_torch, ss = to_torch(
        valid_df, scaler=ss, torch_type=cfg["torch_type"]
    )

    train_dataset = TensorDataset(train_torch)
    valid_dataset = TensorDataset(valid_torch)

    out = {
        "org_data": df,
        "train_data": train_df,
        "train_dataset": train_dataset,
        "valid_data": valid_df,
        "valid_dataset": valid_dataset,
        "scaler": ss,
    }

    return out

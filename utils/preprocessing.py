import pandas as pd


def df_index_resetter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index().drop("index", axis=1)
    return df


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

    out = {
        "org_data": df,
        "train_data": train_df,
        "valid_data": valid_df
    }

    return out

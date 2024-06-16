import torch
from torch.utils.data import DataLoader

from experiment_tools.set_up import start_experiment
from genifyer.model.VariationalAutoEncoder import VAE
from genifyer.trainer.loop import train
from genifyer.trainer.opt import options
from utils.cfg_diff import get_config, get_diff
from utils.preprocessing import make_datasets
from utils.result import plot_data


default_filename = "../../config/default/VAE.json"
exp_filename = "../../config/experiment/VAE.json"

_, cfg, default_str, exp_str = get_config(default_filename, exp_filename)

logger = start_experiment(cfg)
logger = get_diff(default_str, exp_str, logger)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"device: {device}")

out = make_datasets(cfg)

logger.info(f"original data records: {len(out["org_data"])}")
logger.info(f"train data records: {len(out["train_data"])}")
logger.info(f"valid data records: {len(out["valid_data"])}")
logger.info(f"train raw data:\n\n{out["train_data"]}\n")

train_dataloader = DataLoader(
    dataset=out["train_dataset"],
    batch_size=cfg["dataloader_params"]["batch_size"],
    shuffle=cfg["dataloader_params"]["shuffle"],
    sampler=cfg["dataloader_params"]["sampler"],
    batch_sampler=cfg["dataloader_params"]["batch_sampler"],
    num_workers=cfg["dataloader_params"]["num_workers"],
    collate_fn=cfg["dataloader_params"]["collate_fn"],
    pin_memory=cfg["dataloader_params"]["pin_memory"],
    drop_last=cfg["dataloader_params"]["drop_last"],
    timeout=cfg["dataloader_params"]["timeout"],
    worker_init_fn=cfg["dataloader_params"]["worker_init_fn"],
    prefetch_factor=cfg["dataloader_params"]["prefetch_factor"],
    persistent_workers=cfg["dataloader_params"]["persistent_workers"],
    pin_memory_device=cfg["dataloader_params"]["pin_memory_device"]
)
valid_dataloader = DataLoader(
    dataset=out["valid_dataset"],
    batch_size=cfg["dataloader_params"]["batch_size"],
    shuffle=cfg["dataloader_params"]["shuffle"],
    sampler=cfg["dataloader_params"]["sampler"],
    batch_sampler=cfg["dataloader_params"]["batch_sampler"],
    num_workers=cfg["dataloader_params"]["num_workers"],
    collate_fn=cfg["dataloader_params"]["collate_fn"],
    pin_memory=cfg["dataloader_params"]["pin_memory"],
    drop_last=cfg["dataloader_params"]["drop_last"],
    timeout=cfg["dataloader_params"]["timeout"],
    worker_init_fn=cfg["dataloader_params"]["worker_init_fn"],
    prefetch_factor=cfg["dataloader_params"]["prefetch_factor"],
    persistent_workers=cfg["dataloader_params"]["persistent_workers"],
    pin_memory_device=cfg["dataloader_params"]["pin_memory_device"]
)

model = VAE(
    input_dim=cfg["params"]["input_dim"],
    hidden_dim=cfg["params"]["hidden_dim"],
    latent_dim=cfg["params"]["latent_dim"]
).to(device=device)
logger.info(f"model architecture:\n\n{model}\n")
opt = options(cfg, model)
optimizer = opt.getter()
epochs = cfg["params"]["epochs"]

model, training_data = train(
    epochs=epochs,
    train_dataloader=train_dataloader,
    valid_dataloader=valid_dataloader,
    model=model,
    optimizer=optimizer,
    batch_size=cfg["dataloader_params"]["batch_size"],
    device=device
)

out_dir = cfg["log"]["log_file"].replace("VAE.log", "")
plot_data(training_data, out_dir)
torch.save(model.state_dict(), f"{out_dir}model_weight.pth")

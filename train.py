import argparse
from icecube.utils import get_batch_paths
from torch.utils.data import DataLoader
import config
from pdb import set_trace
import os


def train(cfg):
    custom_model = cfg.MODEL_NAME()
    opt = cfg.OPT(
        custom_model.parameters(), lr=cfg.LR, weight_decay=cfg.WD
    )
    loss_func = cfg.LOSS_FUNC()
    scheduler = cfg.SCHEDULER(
        opt,
        num_warmup_steps=cfg.WARM_UP_PCT * cfg.EPOCHS,
        num_training_steps=cfg.EPOCHS,
    )


    cfg.FIT_FUNC(
        epochs=cfg.EPOCHS,
        model=custom_model,
        loss_fn=loss_func,
        opt=opt,
        metric=cfg.METRIC,
        config = cfg,
        folder=cfg.FOLDER/cfg.EXP_NAME,
        exp_name=f"{cfg.EXP_NAME}",
        device=cfg.DEVICE,
        sched=scheduler,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default=None)
    args = parser.parse_args()
    configs = eval(f"config.{args.config_name}")
    print(f"Training with config: {configs.__dict__}")
    #os.makedirs(configs.FOLDER/configs.EXP_NAME)
    train(configs)

if __name__ == "__main__":
    main()

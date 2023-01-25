import argparse
from icecube.utils import get_batch_paths
from torch.utils.data import DataLoader
import config
from pdb import set_trace
from icecube.utils import fit
import os


def train(cfg):
    trn_ds = cfg.TRN_DATASET(
        get_batch_paths(
            cfg.TRN_BATCH_RANGE[0],
            cfg.TRN_BATCH_RANGE[1],
            cache_dir=cfg.DATA_CACHE_DIR,
        )
    )
    vld_ds  = cfg.VAL_DATASET(
        get_batch_paths(
            cfg.VAL_BATCH_RANGE[0],
            cfg.VAL_BATCH_RANGE[1],
            cache_dir=cfg.DATA_CACHE_DIR,
        )
    )
    trn_dl = DataLoader(
        trn_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=cfg.PRESISTENT_WORKERS,
        collate_fn=cfg.COLLAT_FN,
    )
    vld_dl = DataLoader(
        vld_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=cfg.PRESISTENT_WORKERS,
        collate_fn=cfg.COLLAT_FN,
    )

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


    fit(
        epochs=cfg.EPOCHS,
        model=custom_model,
        train_dl=trn_dl,
        valid_dl=vld_dl,
        loss_fn=loss_func,
        opt=opt,
        metric=cfg.METRIC,
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
    os.makedirs(configs.FOLDER/configs.EXP_NAME)
    train(configs)

if __name__ == "__main__":
    main()

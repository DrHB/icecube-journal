import argparse
from icecube.utils import get_batch_paths
from torch.utils.data import DataLoader
import config
from pdb import set_trace
import os
from accelerate import Accelerator
from fastprogress.fastprogress import master_bar, progress_bar
import torch
import pandas as pd
import numpy as np
from pathlib import Path


def fit_accelerate(
    epochs,
    model,
    train_dl,
    valid_dl,
    loss_fn,
    opt,
    metric,
    num_workers,
    folder="models",
    exp_name="exp_00",
    sched=None,
     
):

    os.makedirs(folder, exist_ok=True)

    mb = master_bar(range(epochs))
    mb.write(["epoch", "train_loss", "valid_loss", "val_metric"], table=True)


    accelerator = Accelerator(mixed_precision='fp16')
    accelerator.print(accelerator.device)
    model = model.to(accelerator.device)
    model, opt, train_dl, valid_dl, sched = accelerator.prepare(
        model, opt, train_dl, valid_dl, sched
    )

    for i in mb:  # iterating  epoch
        trn_loss, val_loss = 0.0, 0.0
        model.train()  # set model for training
        for batch in progress_bar(train_dl, parent=mb):
            # putting batches to device
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            out = model(batch['event'], mask=batch['mask'])  # forward pass
            loss = loss_fn(out, batch['label'])  # calulation loss
            trn_loss += loss.item()
            accelerator.backward(loss)
            opt.step()
            sched.step()
            opt.zero_grad()
        trn_loss /= mb.child.total

        # putting model in eval mode
        model.eval()
        gt = []
        pred = []
        # after epooch is done we can run a validation dataloder and see how are doing
        for batch in progress_bar(valid_dl, parent=mb):
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            with torch.no_grad():  # half precision
                out = model(batch['event'], mask=batch['mask'])  # forward pass
                loss = loss_fn(out, batch['label'])  # calulation loss
            val_loss += loss.item()
            predictions, references = accelerator.gather_for_metrics((out, batch["label"]))
            gt.append(references)
            pred.append(predictions)
        # calculating metric
        metric_ = metric(torch.cat(pred), torch.cat(gt))
        # saving model if necessary

        accelerator.save_state(f"{Path(folder)/exp_name}_{i}")
        
        val_loss /= mb.child.total
        res =         pd.DataFrame(
            {
                "epoch": [i],
                "train_loss": [trn_loss],
                "valid_loss": [val_loss],
                "metric": [metric_],
            }
        )
        accelerator.print(res)
        res.to_csv(f"{Path(folder)/exp_name}_{i}.csv", index=False)
    print("Training done")

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

    fit_accelerate(
        epochs=cfg.EPOCHS,
        model=custom_model,
        train_dl=trn_dl,
        valid_dl=vld_dl,
        loss_fn=loss_func,
        opt=opt,
        metric=cfg.METRIC,
        folder=cfg.FOLDER/cfg.EXP_NAME,
        exp_name=f"{cfg.EXP_NAME}",
        num_workers=cfg.NUM_WORKERS,
        sched=scheduler,
    )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default=None)
    args = parser.parse_args()
    configs = eval(f"config.{args.config_name}")
    print(f"Training with config: {configs.__dict__}")
    os.makedirs(configs.FOLDER/configs.EXP_NAME)
    train(configs)

if __name__ == "__main__":
    main()

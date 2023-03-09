import argparse
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as gDataLoader
import config
import os
from accelerate import Accelerator
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk, concatenate_datasets
import random

def run_eval(dl, model, acc, loss_fn, metric):
        model.eval()
        gt = []
        pred = []
        # after epooch is done we can run a validation dataloder and see how are doing
        for batch in (dl):
            batch = batch.to(acc.device)
            with torch.no_grad():  # half precision
                out = model(batch)  # forward pass
                loss = loss_fn(out, batch.y)  # calulation loss
            val_loss += loss.item()
            predictions, references = acc.gather_for_metrics((out, batch.y))
            gt.append(references)
            pred.append(predictions)
        # calculating metric
        metric_ = metric(torch.cat(pred), torch.cat(gt))
        # saving model if necessary
        return metric_, val_loss
    
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
    gtrad_acc_steps = 4, 
     
):

    os.makedirs(folder, exist_ok=True)
    accelerator = Accelerator(mixed_precision='fp16',
                              gradient_accumulation_steps=gtrad_acc_steps)
    accelerator.print(accelerator.device)
    model = model.to(accelerator.device)
    valid_iter = len(valid_dl)//8
    model, opt, train_dl, valid_dl, sched = accelerator.prepare(
        model, opt, train_dl, valid_dl, sched
    )

    for epoch in range(epochs):
        trn_loss, val_loss = 0.0, 0.0
        model.train()  # set model for training
        for step, batch in tqdm(enumerate(train_dl)):
            # putting batches to device
            batch = batch.to(accelerator.device)
            with accelerator.accumulate(model):
                out = model(batch)  # forward pass
                loss = loss_fn(out, batch.y)  # calulation loss
                trn_loss += loss.item()
                accelerator.backward(loss)
                opt.step()
                sched.step()
                opt.zero_grad()
            
            if step % valid_iter == 0:
                metric_, val_loss = run_eval(valid_dl, model, accelerator, loss_fn, metric)
                accelerator.save_state(f"{Path(folder)/exp_name}_{step}")
                val_loss /= len(valid_dl)
                res = pd.DataFrame(
                    {
                        "epoch": [i],
                        "train_loss": [trn_loss/step],
                        "valid_loss": [val_loss/len(valid_dl)],
                        "metric": [metric_],
                    }
                )
                accelerator.print(res)
                res.to_csv(f"{Path(folder)/exp_name}_{i}.csv", index=False)
                model.train()


def train(cfg):
    
    vld_pth = [
            load_from_disk(cfg.DATA_CACHE_DIR / f"batch_{i}.parquet")
            for i in range(cfg.VAL_BATCH_RANGE[0], cfg.VAL_BATCH_RANGE[1])
        ]
    vld_pth = concatenate_datasets(vld_pth)
    vld_ds = cfg.VAL_DATASET(vld_pth)

    nums = [i for i in range(cfg.TRN_BATCH_RANGE[0], cfg.TRN_BATCH_RANGE[1])]
    random.shuffle(nums)
    trn_pth = [
                load_from_disk(cfg.DATA_CACHE_DIR / f"batch_{i}.parquet") for i in nums
            ]
    trn_pth = concatenate_datasets(trn_pth)
    trn_ds = cfg.TRN_DATASET(trn_pth)

    trn_dl = gDataLoader(
        trn_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=cfg.PRESISTENT_WORKERS,
        collate_fn=cfg.COLLAT_FN,
    )
    vld_dl = gDataLoader(
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
        num_warmup_steps=1000,
        num_training_steps=len(trn_dl) * cfg.EPOCHS,
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
        gtrad_acc_steps=cfg.GRAD_ACC_STEPS
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

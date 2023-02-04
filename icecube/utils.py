# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_utils.ipynb.

# %% auto 0
__all__ = ['label_to_df', 'get_size', 'reduce_mem_usage', 'get_config_as_dict', 'save_folder', 'save_pred_as_csv', 'SaveModel',
           'SaveModelMetric', 'SaveModelEpoch', 'fit', 'fit_shuflle', 'compare_events', 'get_batch_paths',
           'angular_dist_score', 'get_score', 'get_score_vector', 'collate_fn', 'collate_fn_v1', 'collate_fn_graphv0',
           'good_luck']

# %% ../nbs/00_utils.ipynb 1
import numpy as np
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from fastprogress.fastprogress import master_bar, progress_bar
from typing import List
from accelerate import Accelerator
import gc
from datasets import load_dataset, load_from_disk, concatenate_datasets
import random
import wandb

# %% ../nbs/00_utils.ipynb 2
def label_to_df(label, angle_post_fix = '', vec_post_fix = ''):
    df = pd.DataFrame(label, columns=['direction_x', 'direction_y', 'direction_z'])
    r = np.sqrt(df['direction_x'+ vec_post_fix]**2 + df['direction_y'+ vec_post_fix]**2 + df['direction_z' + vec_post_fix]**2)
    df['zenith' + angle_post_fix] = np.arccos(df['direction_z'+ vec_post_fix]/r)
    df['azimuth'+ angle_post_fix] = np.arctan2(df['direction_y'+ vec_post_fix],df['direction_x' + vec_post_fix]) #np.sign(results['true_y'])*np.arccos((results['true_x'])/(np.sqrt(results['true_x']**2 + results['true_y']**2)))
    df['azimuth'+ angle_post_fix][df['azimuth'  + angle_post_fix]<0] = df['azimuth'  + angle_post_fix][df['azimuth'  +  angle_post_fix]<0] + 2*np.pi 
    return df

def get_size(df):
    return round(df.memory_usage(deep=True).sum() / 1024 ** 3, 2)


def reduce_mem_usage(df):
    start_mem = get_size(df)
    print(f"Memory usage of dataframe is {start_mem} GB")
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif (
                    c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max
                ):
                    df[col] = df[col].astype(np.int16)
                elif (
                    c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max
                ):
                    df[col] = df[col].astype(np.int32)
                elif (
                    c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max
                ):
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = get_size(df)
    print(f"Memory usage after optimization is: {end_mem} GB")
    print(f"Decreased by {100 * (start_mem - end_mem) / start_mem}%")
    return df

def get_config_as_dict(config_name):
    out = dict()
    for att in dir(config_name):
        if att.isupper():
            out[att] = getattr(config_name, att)
    return out


# %% ../nbs/00_utils.ipynb 4
def save_folder(BATCH: int, CFG, n_jobs: int = 48):
    """
    Save the events in a folder


    """
    print("...")
    print("Loading data...")
    train_meta = pd.read_parquet(CFG.PATH_META).query("batch_id == @BATCH")
    geometry = pd.read_csv(CFG.PATH_GEOMETRY)
    train = pd.read_parquet(CFG.PATH_DATASET / "train" / f"batch_{BATCH}.parquet")

    # current save path
    CURRENT_SAVE_PATH = CFG.SAVE_PATH / f"batch_{BATCH}"
    os.makedirs(CURRENT_SAVE_PATH, exist_ok=True)

    # function to get the event from the dataframe
    def get_event(event_id: int):
        one_event = train_meta.query("event_id == @event_id")
        event = train.loc[event_id]
        event = event.merge(geometry, on="sensor_id")
        target = one_event[["azimuth", "zenith"]]
        return {"event": event.to_records(), "target": target.to_records()}

    # function that saves event as .pth file
    def save_event(event_id: int, save_path: Path):
        event = get_event(event_id)
        torch.save(event, save_path / f"{event_id}.pth")

    # function that save in parallel all the using joblib
    def save_all_events(event_ids: int, save_path: Path, n_jobs: int = 8):
        Parallel(n_jobs=n_jobs)(
            delayed(save_event)(event_id, save_path) for event_id in tqdm(event_ids)
        )

    print("Saving data for batch", BATCH)
    save_all_events(train_meta.event_id.unique(), CURRENT_SAVE_PATH, n_jobs=n_jobs)

def save_pred_as_csv(y_hat: torch.Tensor, y: torch.Tensor, name: Path):
    y_hat = y_hat.cpu().numpy()
    y = y.cpu().numpy()
    azimuth_gt = y[:, 0]
    zenith_gt = y[:, 1]
    azimuth_pred = y_hat[:, 0]
    zenith_pred = y_hat[:, 1]
    df = pd.DataFrame(
        {
            "azimuth_gt": azimuth_gt,
            "zenith_gt": zenith_gt,
            "azimuth_pred": azimuth_pred,
            "zenith_pred": zenith_pred,
        }
    )
    df.to_csv(f"{name}.csv", index=False)


# %% ../nbs/00_utils.ipynb 5
class SaveModel:
    def __init__(self, folder, exp_name, best=np.inf):
        self.best = best
        self.folder = Path(folder) / f"{exp_name}.pth"

    def __call__(self, score, model, epoch):
        if score < self.best:
            self.best = score
            print(f"Better model found at epoch {epoch} with value: {self.best}.")
            torch.save(model.state_dict(), self.folder)


class SaveModelMetric:
    def __init__(self, folder, exp_name, best=-np.inf):
        self.best = best
        self.folder = Path(folder) / f"{exp_name}.pth"

    def __call__(self, score, model, epoch):
        if score > self.best:
            self.best = score
            print(f"Better model found at epoch {epoch} with value: {self.best}.")
            torch.save(model.state_dict(), self.folder)


class SaveModelEpoch:
    def __init__(self, folder, exp_name, best=-np.inf):
        self.best = best
        self.folder = Path(folder)
        self.exp_name = exp_name

    def __call__(self, score, model, epoch):
        self.best = score
        print(f"Better model found at epoch {epoch} with value: {self.best}.")
        torch.save(model.state_dict(), f"{self.folder/self.exp_name}_{epoch}.pth")


def fit(
    epochs,
    model,
    train_dl,
    valid_dl,
    loss_fn,
    opt,
    metric,
    folder="models",
    exp_name="exp_00",
    device=None,
    sched=None,
    save_md=SaveModelEpoch,
):
    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    os.makedirs(folder, exist_ok=True)

    mb = master_bar(range(epochs))
    mb.write(["epoch", "train_loss", "valid_loss", "val_metric"], table=True)
    model.to(device)  # we have to put our model on gpu
    scaler = torch.cuda.amp.GradScaler()  # this for half precision training
    save_md = save_md(folder, exp_name)

    for i in mb:  # iterating  epoch
        trn_loss, val_loss = 0.0, 0.0
        trn_n, val_n = len(train_dl.dataset), len(valid_dl.dataset)
        model.train()  # set model for training
        for batch in progress_bar(train_dl, parent=mb):
            # putting batches to device
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast():  # half precision
                out = model(batch["event"], mask=batch["mask"])  # forward pass
                loss = loss_fn(out, batch["label"])  # calulation loss

            trn_loss += loss.item()

            scaler.scale(loss).backward()  # backward
            scaler.step(opt)  # optimzers step
            scaler.update()  # for half precision
            opt.zero_grad()  # zeroing optimizer
            if sched is not None:
                sched.step()  # scuedular step

        trn_loss /= mb.child.total

        # putting model in eval mode
        model.eval()
        gt = []
        pred = []
        # after epooch is done we can run a validation dataloder and see how are doing
        with torch.no_grad():
            for batch in progress_bar(valid_dl, parent=mb):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.cuda.amp.autocast():  # half precision
                    out = model(batch["event"], mask=batch["mask"])  # forward pass
                    loss = loss_fn(out, batch["label"])  # calulation loss
                val_loss += loss.item()

                gt.append(batch["label"].detach())
                pred.append(out.detach())
        # calculating metric
        metric_ = metric(torch.cat(pred), torch.cat(gt))
        # saving model if necessary
        save_md(metric_, model, i)
        val_loss /= mb.child.total
        res = pd.DataFrame(
            {
                "epoch": [i],
                "train_loss": [trn_loss],
                "valid_loss": [val_loss],
                "metric": [metric_],
            }
        )
        print(res)
        res.to_csv(f"{Path(folder)/exp_name}_{i}.csv", index=False)
        gc.collect()
    print("Training done")


def fit_shuflle(
    epochs,
    model,
    loss_fn,
    opt,
    metric,
    config,
    folder="models",
    exp_name="exp_00",
    device=None,
    sched=None,
    save_md=SaveModelEpoch,
):
    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    os.makedirs(folder, exist_ok=True)

    mb = master_bar(range(epochs))
    mb.write(["epoch", "train_loss", "valid_loss", "val_metric"], table=True)
    model.to(device)  # we have to put our model on gpu
    scaler = torch.cuda.amp.GradScaler()  # this for half precision training
    save_md = save_md(folder, exp_name)

    vld_pth = [
        load_from_disk(config.DATA_CACHE_DIR / f"batch_{i}.parquet")
        for i in range(config.VAL_BATCH_RANGE[0], config.VAL_BATCH_RANGE[1])
    ]

    vld_pth = concatenate_datasets(vld_pth)

    vld_ds = config.VAL_DATASET(vld_pth)

    valid_dl = DataLoader(
        vld_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=config.PRESISTENT_WORKERS,
        collate_fn=config.COLLAT_FN,
    )
    
    wandb.init(
        project="ice",
        entity="kaggle-hi",
        name=config.EXP_NAME,
        config=get_config_as_dict(config),
    )
    wandb.watch(model)

    for i in mb:  # iterating  epoch
        trn_loss, val_loss = 0.0, 0.0
        # shuffling the data before every epoch cheaper than shuffling the dataloader

        nums = [i for i in range(config.TRN_BATCH_RANGE[0], config.TRN_BATCH_RANGE[1])]
        random.shuffle(nums)
        trn_pth = [
            load_from_disk(config.DATA_CACHE_DIR / f"batch_{i}.parquet") for i in nums
        ]

        trn_pth = concatenate_datasets(trn_pth)
        trn_ds = config.TRN_DATASET(trn_pth)

        train_dl = DataLoader(
            trn_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=config.PRESISTENT_WORKERS,
            collate_fn=config.COLLAT_FN,
        )

        trn_n, val_n = len(train_dl.dataset), len(valid_dl.dataset)
        model.train()  # set model for training
        for batch in progress_bar(train_dl, parent=mb):
            # putting batches to device
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast():  # half precision
                out = model(batch)  # forward pass
                loss = loss_fn(out, batch["label"])  # calulation loss

            trn_loss += loss.item()

            scaler.scale(loss).backward()  # backward
            scaler.step(opt)  # optimzers step
            scaler.update()  # for half precision
            opt.zero_grad()  # zeroing optimizer
            if sched is not None:
                sched.step()  # scuedular step

        trn_loss /= mb.child.total

        # putting model in eval mode
        model.eval()
        gt = []
        pred = []
        # after epooch is done we can run a validation dataloder and see how are doing
        with torch.no_grad():
            for batch in progress_bar(valid_dl, parent=mb):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.cuda.amp.autocast():  # half precision
                    out = model(batch)  # forward pass
                    loss = loss_fn(out, batch["label"])  # calulation loss
                val_loss += loss.item()

                gt.append(batch["label"].detach())
                pred.append(out.detach())
        # calculating metric
        metric_ = metric(torch.cat(pred), torch.cat(gt))
        # saving predictions
        #save_pred_as_csv(
        #    torch.cat(pred), torch.cat(gt), name=f"{Path(folder)/exp_name}_OOF_{i}.csv"
        #)
        # saving model if necessary
        save_md(metric_, model, i)

        val_loss /= mb.child.total

        wandb.log(
            {
                "epoch": i,
                "train_loss": trn_loss,
                "valid_loss": val_loss,
                "metric": metric_,
            }
        )

        res = pd.DataFrame(
            {
                "epoch": [i],
                "train_loss": [trn_loss],
                "valid_loss": [val_loss],
                "metric": [metric_],
            }
        )
        print(res)
        res.to_csv(f"{Path(folder)/exp_name}_{i}.csv", index=False)
        gc.collect()
    print("Training done")


# %% ../nbs/00_utils.ipynb 8
#function that compares events from parquet and pth files
def compare_events(event_id: int, CFG, BATCH: int):
    train_meta = pd.read_parquet(CFG.PATH_META)
    geometry = pd.read_csv(CFG.PATH_GEOMETRY)
    train = pd.read_parquet(CFG.PATH_DATASET/'train'/f'batch_{BATCH}.parquet')
    one_event = train_meta.query("event_id == @event_id")
    event = train.loc[event_id]
    event = event.merge(geometry, on="sensor_id")
    event_pth = pd.DataFrame.from_records(torch.load(CFG.SAVE_PATH/f'batch_{BATCH}' / f"{event_id}.pth")['event']).iloc[:, 1:]
    return np.all(event == event_pth)

# %% ../nbs/00_utils.ipynb 9
def get_batch_paths(
    start: int,
    end: int,
    extension: str = "*.pth",
    cache_dir: Path = Path("../data/cache"),
) -> List[Path]:
    """Get paths to all files in a range of batches"""
    trn_path = []
    for i in range(start, end + 1):
        path = (cache_dir / f"batch_{i}").glob(extension)
        trn_path.extend(list(path))
    return trn_path


def angular_dist_score(
    az_true: torch.Tensor,
    zen_true: torch.Tensor,
    az_pred: torch.Tensor,
    zen_pred: torch.Tensor,
) -> torch.Tensor:
    sa1 = torch.sin(az_true)
    ca1 = torch.cos(az_true)
    sz1 = torch.sin(zen_true)
    cz1 = torch.cos(zen_true)

    sa2 = torch.sin(az_pred)
    ca2 = torch.cos(az_pred)
    sz2 = torch.sin(zen_pred)
    cz2 = torch.cos(zen_pred)

    scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + cz1 * cz2
    scalar_prod = torch.clamp(scalar_prod, -1, 1)
    return torch.mean(torch.abs(torch.acos(scalar_prod)))


# calculte metric based on angular distance
def get_score(y_hat, y):
    return (
        angular_dist_score(y[:, 0], y[:, 1], y_hat[:, 0], y_hat[:, 1])
        .detach()
        .cpu()
        .numpy()
    )


def get_score_vector(y_hat, y):
    y_hat = label_to_df(y_hat.detach().cpu().numpy()[:, :3])
    y = label_to_df(y.detach().cpu().numpy())

    return (
        angular_dist_score(
            torch.tensor(y["azimuth"].values, dtype=torch.float32),
            torch.tensor(y["zenith"].values, dtype=torch.float32),
            torch.tensor(y_hat["azimuth"].values, dtype=torch.float32),
            torch.tensor(y_hat["zenith"].values, dtype=torch.float32),
        )
        .detach()
        .cpu()
        .numpy()
    )


def collate_fn(batch):
    event = [x["event"] for x in batch]
    mask = [x["mask"] for x in batch]
    label = [x["label"] for x in batch]

    event = torch.nn.utils.rnn.pad_sequence(event, batch_first=True)
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True)
    batch = {"event": event, "mask": mask, "label": torch.stack(label)}
    return batch


def collate_fn_v1(batch):

    event = [x["event"] for x in batch]
    mask = [x["mask"] for x in batch]
    label = [x["label"] for x in batch]
    sensor_id = [x["sensor_id"] for x in batch]

    event = torch.nn.utils.rnn.pad_sequence(event, batch_first=True)
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True)
    sensor_id = torch.nn.utils.rnn.pad_sequence(sensor_id, batch_first=True)
    batch = {
        "event": event,
        "mask": mask,
        "label": torch.stack(label),
        "sensor_id": sensor_id,
    }
    return batch


def collate_fn_graphv0(batch):

    event = [x["event"] for x in batch]
    mask = [x["mask"] for x in batch]
    label = [x["label"] for x in batch]
    distance_matrix = [x["distance_matrix"] for x in batch]
    adjecent_matrix = [x["adjecent_matrix"] for x in batch]

    event = torch.nn.utils.rnn.pad_sequence(event, batch_first=True)
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True)

    batch = {
        "event": event,
        "mask": mask,
        "label": torch.stack(label),
        "distance_matrix": torch.stack(distance_matrix),
        "adjecent_matrix": torch.stack(adjecent_matrix),
    }
    return batch


# %% ../nbs/00_utils.ipynb 13
def good_luck():
    return True

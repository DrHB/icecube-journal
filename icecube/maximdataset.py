# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_maximdataset.ipynb.

# %% auto 0
__all__ = ['PATH', 'flatten', 'get_edge_index', 'RandomChunkSampler', 'LenMatchBatchSampler', 'prepare_sensors',
           'ice_transparency', 'IceCubeDataset', 'IceCubeDataset_len', 'dict_to', 'to_device', 'DeviceDataLoader',
           'WrapperAdamW', 'get_dataloaders']

# %% ../nbs/00_maximdataset.ipynb 3
import polars as pl
import pandas as pd
import gc
import os
import numpy as np
import torch
from typing import Iterator, Optional, Sized
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import polars as pl
import pandas as pd
import os,gc
import numpy as np
from collections import OrderedDict
from sklearn.preprocessing import RobustScaler
from scipy.interpolate import interp1d
from fastai.vision.all import DataLoaders, OptimWrapper
from torch_geometric.nn.pool import knn_graph
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

# %% ../nbs/00_maximdataset.ipynb 7
PATH = '/opt/slh/icecube/data/'

def flatten(o):
    "Concatenate all collections and items as a generator"
    for item in o:
        if isinstance(o, dict): yield o[item]; continue
        elif isinstance(item, str): yield item; continue
        try: yield from flatten(item)
        except TypeError: yield item
        
        
def get_edge_index(pos, time, mask, L, k =8):
    xyzt = torch.concat([pos, time.view(-1, 1)], dim=1)[mask]
    edge_index = knn_graph(
            xyzt[:, [0, 1, 2,3]],  # x, y, z
            k=k,
            batch=None,
            loop=False
        )
    
    total_len = L * k
    to_pad = total_len - edge_index.shape[1]
    edge_mask = torch.zeros(2, total_len, dtype=torch.bool)
    
    if to_pad > 0:
        edge_mask[:, :edge_index.shape[1]] = True
        edge_index = F.pad(edge_index, (0, to_pad), "constant", 0) 
    else:
        edge_mask[:] = True
    return edge_index, edge_mask
        
class RandomChunkSampler(torch.utils.data.Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, num_samples: Optional[int] = None,
                 generator=None, chunk_size=200000, **kwargs) -> None:
        self.data_source = data_source
        self._num_samples = num_samples
        self.generator = generator
        self.chunk_size = chunk_size

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        chunk_list = torch.randperm(self.num_samples // self.chunk_size, generator=generator).tolist()
        for i in range(self.num_samples // self.chunk_size):
            chunk = chunk_list[i]
            yield from (chunk*self.chunk_size + torch.randperm(self.chunk_size, generator=generator)).tolist()
        #yield from ((self.num_samples // self.chunk_size)*self.chunk_size + 
        #    torch.randperm(self.num_samples%self.chunk_size, generator=generator)).tolist()

    def __len__(self) -> int:
        return self.num_samples
    
class LenMatchBatchSampler(torch.utils.data.BatchSampler):
    def __iter__(self):
        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            s = self.sampler.data_source[idx]
            if isinstance(s,tuple): L = s[0]["mask"].sum()
            else: L = s["mask"].sum()
            #if torch.rand(1).item() < 0.1: L = int(1.5*L)
            L = L // 16 
            if len(buckets[L]) == 0:  buckets[L] = []
            buckets[L].append(idx)
            
            if len(buckets[L]) == self.batch_size:
                batch = list(buckets[L])
                yield batch
                yielded += 1
                buckets[L] = []
                
        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch
            
def prepare_sensors(path=PATH):
    sensors = pd.read_csv(os.path.join(path,'sensor_geometry.csv')).astype(
        {
            "sensor_id": np.int16,
            "x": np.float32,
            "y": np.float32,
            "z": np.float32,
        }
    )
    sensors["string"] = 0
    sensors["qe"] = 0#1

    for i in range(len(sensors) // 60):
        start, end = i * 60, (i * 60) + 60
        sensors.loc[start:end, "string"] = i

        # High Quantum Efficiency in the lower 50 DOMs - https://arxiv.org/pdf/2209.03042.pdf (Figure 1)
        if i in range(78, 86):
            start_veto, end_veto = i * 60, (i * 60) + 10
            start_core, end_core = end_veto + 1, (i * 60) + 60
            sensors.loc[start_core:end_core, "qe"] = 1# 1.35

    # https://github.com/graphnet-team/graphnet/blob/b2bad25528652587ab0cdb7cf2335ee254cfa2db/src/graphnet/models/detector/icecube.py#L33-L41
    # Assume that "rde" (relative dom efficiency) is equivalent to QE
    sensors["x"] /= 500
    sensors["y"] /= 500
    sensors["z"] /= 500
    #sensors["qe"] -= 1.25
    #sensors["qe"] /= 0.25

    return sensors

def ice_transparency(path=PATH, datum=1950):
    # Data from page 31 of https://arxiv.org/pdf/1301.5361.pdf
    # Datum is from footnote 8 of page 29
    df = pd.read_csv(os.path.join(path,'ice_transparency.txt'), delim_whitespace=True)
    df["z"] = df["depth"] - datum
    df["z_norm"] = df["z"] / 500
    df[["scattering_len_norm", "absorption_len_norm"]] = RobustScaler().fit_transform(
        df[["scattering_len", "absorption_len"]])

    # These are both roughly equivalent after scaling
    f_scattering = interp1d(df["z_norm"], df["scattering_len_norm"])
    f_absorption = interp1d(df["z_norm"], df["absorption_len_norm"])
    return f_scattering, f_absorption

class IceCubeDataset(Dataset):
    def __init__(self, path=PATH, chunk_size=200000, L=256, buf_size=4, train=True, reduce_size=-1):
        #path_geometry=PATH_GEOMETRY, /sensor_geometry.csv
        self.path = os.path.join(path,'train')
        self.files = [p for p in sorted(os.listdir(self.path)) if p!='batch_660.parquet'] #660 is shorter
        val_fnames = ['batch_655.parquet','batch_656.parquet','batch_657.parquet','batch_658.parquet',
                      'batch_659.parquet']
        if not train: self.files = val_fnames
        else: self.files = sorted(set(self.files) - set(val_fnames))
        self.chunk_size = chunk_size
        self.buf = OrderedDict()
        self.L,self.buf_size = L,buf_size
        sensors = prepare_sensors(path)
        self.geometry = torch.from_numpy(sensors[['x','y','z']].values.astype(np.float32))
        self.qe = sensors['qe'].values
        self.ice_properties = ice_transparency(path)
        self.train = train
        
        df = pd.read_parquet(os.path.join(path,'train_meta.parquet'))
        df = df[['event_id','azimuth','zenith']]
        df['azimuth'] = df['azimuth'].astype(np.float32)
        df['zenith'] = df['zenith'].astype(np.float32)
        df['event_id'] = df['event_id'].astype(np.int32)
        df = df.set_index('event_id',drop=True)
        self.target = df
        gc.collect()
        self.reduce_size = reduce_size

        
    def __len__(self):
        return len(self.files)*self.chunk_size if self.reduce_size < 0 \
                else int(self.reduce_size*len(self.files))*self.chunk_size
        
    def __getitem__(self, idx0):
        fname = self.files[idx0//self.chunk_size]
        if fname not in self.buf:
            df = pl.read_parquet(os.path.join(self.path,fname))
            df = df.groupby("event_id").agg([
                pl.count(),
                pl.col("sensor_id").list(),
                pl.col("time").list(),
                pl.col("charge").list(),
                pl.col("auxiliary").list(),])
            self.buf[fname] = df.sort('event_id')
            if len(self.buf) > self.buf_size: del self.buf[list(self.buf.keys())[0]]
        
        idx = idx0%self.chunk_size
        df = self.buf[fname]
        sensor_id =  df[idx]['sensor_id'][0].item().to_numpy()
        time =  df[idx]['time'][0].item().to_numpy()
        charge = df[idx]['charge'][0].item().to_numpy()
        auxiliary = df[idx]['auxiliary'][0].item().to_numpy()
        event_idx = df[idx]['event_id'].item()
        
        if self.train and np.random.rand() < 0.9:
            print(time.shape[0])
            filter_mask = generate_mask(time.shape[0])
            sensor_id =  sensor_id[filter_mask]
            time =  time[filter_mask]
            charge = charge[filter_mask]
            auxiliary = auxiliary[filter_mask]

            
        #sensor_id = sensor_id[~auxiliary]
        #time = time[~auxiliary]
        #charge = charge[~auxiliary]
        
        time = (time - 1e4)/3e4
        charge = np.log10(charge)/3.0 #np.log(charge)
        
        L = len(sensor_id)
        if L < self.L:
            sensor_id = np.pad(sensor_id,(0,max(0,self.L-L)))
            time = np.pad(time,(0,max(0,self.L-L)))
            charge = np.pad(charge,(0,max(0,self.L-L)))
            auxiliary = np.pad(auxiliary,(0,max(0,self.L-L)))
        else:
            ids = torch.randperm(L).numpy()
            auxiliary_n = np.where(~auxiliary)[0]
            auxiliary_p = np.where(auxiliary)[0]
            ids_n = ids[auxiliary_n][:min(self.L,len(auxiliary_n))]
            ids_p = ids[auxiliary_p][:min(self.L-len(ids_n),len(auxiliary_p))]
            ids = np.concatenate([ids_n,ids_p])
            ids.sort()
            L = len(ids)
            
            sensor_id = sensor_id[ids]
            time = time[ids]
            charge = charge[ids]
            auxiliary = auxiliary[ids]
            L = len(ids)
            
        attn_mask = torch.zeros(self.L, dtype=torch.bool)
        attn_mask[:L] = True
        sensor_id = torch.from_numpy(sensor_id).long()
        pos = self.geometry[sensor_id]
        pos[L:] = 0
        qe = self.qe[sensor_id]
        qe[L:] = 0
        ice_properties = np.stack([self.ice_properties[0](pos[:L,2]),
                                   self.ice_properties[1](pos[:L,2])],-1)
        ice_properties = np.pad(ice_properties,((0,max(0,self.L-L)),(0,0)))
        ice_properties = torch.from_numpy(ice_properties).float()
        
        target = self.target.loc[event_idx].values
        edge_index, edge_mask = get_edge_index(pos, torch.from_numpy(time).float(), attn_mask, L=self.L)

        return {'sensor_id': sensor_id, 
                'time': torch.from_numpy(time).float(),
                'charge': torch.from_numpy(charge).float(), 
                'pos':pos,
                'mask':attn_mask,
                'idx':event_idx,
                'auxiliary':torch.from_numpy(auxiliary).long(),
                'qe':qe, 
                'ice_properties':ice_properties,
                "edge_index": edge_index, 
                'edge_mask': edge_mask}, {'target': torch.from_numpy(target).float() }
    
    
class IceCubeDataset_len(Dataset):
    def __init__(self, path=PATH, chunk_size=200000, L=256, buf_size=2, train=True, reduce_size=-1):
        #path_geometry=PATH_GEOMETRY, /sensor_geometry.csv
        self.path = os.path.join(path,'train')
        self.files = [p for p in sorted(os.listdir(self.path)) if p!='batch_660.parquet'] #660 is shorter
        val_fnames = ['batch_655.parquet','batch_656.parquet','batch_657.parquet','batch_658.parquet',
                      'batch_659.parquet']
        if not train: self.files = val_fnames
        else: self.files = sorted(set(self.files) - set(val_fnames))
        self.chunk_size = chunk_size
        self.buf = OrderedDict()
        self.L,self.buf_size = L,buf_size
        sensors = prepare_sensors(path)
        self.geometry = torch.from_numpy(sensors[['x','y','z']].values.astype(np.float32))
        self.qe = sensors['qe'].values
        self.ice_properties = ice_transparency(path)
        
        gc.collect()
        self.reduce_size = reduce_size
        
    def __len__(self):
        return len(self.files)*self.chunk_size if self.reduce_size < 0 \
                else int(self.reduce_size*len(self.files))*self.chunk_size
        
    def __getitem__(self, idx0):
        fname = self.files[idx0//self.chunk_size]
        if fname not in self.buf:
            df = pl.read_parquet(os.path.join(self.path,fname))
            df = df.groupby("event_id").agg([
                pl.count(),
                pl.col("sensor_id").list(),
                pl.col("time").list(),
                pl.col("charge").list(),
                pl.col("auxiliary").list(),])
            self.buf[fname] = df.sort('event_id')
            if len(self.buf) > self.buf_size: del self.buf[list(self.buf.keys())[0]]
        
        idx = idx0%self.chunk_size
        df = self.buf[fname]
        sensor_id =  df[idx]['sensor_id'][0].item().to_numpy()
        mask = torch.ones(min(len(sensor_id),self.L), dtype=torch.long)
        return {'mask':mask},{}
    
    
def dict_to(x, device='cpu'):
    return {k:x[k].to(device) for k in x}

def to_device(x, device='cpu'):
    return tuple(dict_to(e,device) for e in x)

class DeviceDataLoader:
    def __init__(self, dataloader, device='cpu'):
        self.dataloader = dataloader
        self.device = device
    
    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(dict_to(x, self.device) for x in batch)
            
def WrapperAdamW(param_groups,**kwargs):
    return OptimWrapper(param_groups,torch.optim.AdamW)

def get_dataloaders(bs, L=192, NUM_WORKERS = 4, SEED = 2023, reduce_size=0.125):
    ds_train = IceCubeDataset(train=True,
                              reduce_size=reduce_size,
                              L=L)
    ds_train_len = IceCubeDataset_len(train=True, 
                                      reduce_size=reduce_size,
                                      L=L)
    len_sampler_train = LenMatchBatchSampler(
        RandomChunkSampler(ds_train_len),
        batch_size=bs, 
        drop_last=True)
    dl_train = DeviceDataLoader(DataLoader(ds_train, 
                                           batch_sampler=len_sampler_train, 
                                           num_workers=NUM_WORKERS, 
                                            persistent_workers=True))
    ds_val = IceCubeDataset(train=False, L=L)
    ds_val_len = IceCubeDataset_len(train=False, L=L)
    len_sampler_val = LenMatchBatchSampler(
                RandomChunkSampler(ds_val_len),
                batch_size=bs, 
                drop_last=False)
    dl_val = DeviceDataLoader(DataLoader(ds_val, batch_sampler=len_sampler_val, num_workers=0))

    data = DataLoaders(dl_train,dl_val)
    return data


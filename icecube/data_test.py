import polars as pl
import pandas as pd
import gc, os, random, math
import numpy as np
from tqdm.notebook import tqdm
from collections import OrderedDict
from bisect import bisect_right

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class LenMatchBatchSampler(torch.utils.data.BatchSampler):
    def __iter__(self):
        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            s = self.sampler.data_source[idx]
            if isinstance(s,tuple): L = s[0]["mask"].sum()
            else: L = s["mask"].sum()
            #if torch.rand(1).item() < 0.1: L = int(1.5*L)
            L = max(1,L // 16) 
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
            
from sklearn.preprocessing import RobustScaler
from scipy.interpolate import interp1d

def prepare_sensors(path):
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

def ice_transparency(path, datum=1950):
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
    def __init__(self, path, path_ice_properties, L=256, cache_size=5):
        self.path = os.path.join(path,'test')
        self.files = [p for p in sorted(os.listdir(self.path))]
            
        #make sure that all files are considered regardless the number of events
        self.chunks = []
        for fname in self.files:
            ids = pl.read_parquet(os.path.join(self.path,fname)\
                    ).select(['event_id']).unique().to_numpy().reshape(-1)
            self.chunks.append(len(ids))
        gc.collect()

        self.chunk_cumsum = np.cumsum(self.chunks)
        self.cache = None
        self.L,self.cache_size = L,cache_size
        sensors = prepare_sensors(path)
        self.geometry = torch.from_numpy(sensors[['x','y','z']].values.astype(np.float32))
        self.qe = sensors['qe'].values
        self.ice_properties = ice_transparency(path_ice_properties)
        
    def __len__(self):
        return self.chunk_cumsum[-1]
    
    def load_data(self, fname):
        if self.cache is None: self.cache = OrderedDict()
        if fname not in self.cache:
            df = pl.read_parquet(os.path.join(self.path,fname))
            df = df.groupby("event_id").agg([
                pl.count(),
                pl.col("sensor_id").list(),
                pl.col("time").list(),
                pl.col("charge").list(),
                pl.col("auxiliary").list(),])
            self.cache[fname] = df.sort('event_id')
            if len(self.cache) > self.cache_size: del self.cache[list(self.cache.keys())[0]]
       
    def __getitem__(self, idx0):
        fidx = bisect_right(self.chunk_cumsum, idx0)
        fname = self.files[fidx]
        idx = int(idx0 - self.chunk_cumsum[fidx] + self.chunk_cumsum[0])
        
        self.load_data(fname)
        df = self.cache[fname][idx]
        sensor_id =  df['sensor_id'][0].item().to_numpy()
        time =  df['time'][0].item().to_numpy()
        charge = df['charge'][0].item().to_numpy()
        auxiliary = df['auxiliary'][0].item().to_numpy()
        event_idx = df['event_id'].item()
        
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

        return {'sensor_id': sensor_id, 'time': torch.from_numpy(time).float(),
                'charge': torch.from_numpy(charge).float(), 'pos':pos, 'mask':attn_mask,
                'idx':event_idx, 'auxiliary':torch.from_numpy(auxiliary).long(),
                'qe':qe, 'ice_properties':ice_properties}
    
def dict_to(x, device='cuda'):
    return {k:x[k].to(device) for k in x}

def get_val(pred):
    pred = F.normalize(pred.float(),dim=-1)
    zen = torch.acos(pred[:,2].clip(-1,1))
    f = F.normalize(pred[:,:2],dim=-1)
    az = torch.asin(f[:,0].clip(-1,1))
    az = torch.where(f[:,1] > 0, az, math.pi - az)
    az = torch.where(az > 0, az, az + 2.0*math.pi)
    return torch.stack([az,zen],-1)
    


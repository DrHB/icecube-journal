### EXPERIMENTS

# Baseline 
#make a table

| EXP_NAME | SCORE     | DESCRIPTION                                                                                                                        | SCRIPT                                        | TRN_SET |
| :------- | :-------- | :--------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------- |:-------------------------------------------- |
| EXP_00   | `1.182` | Baseline experiment, `6` Blocks of Transformer Encoder, number of events is restricted to 100                                        | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_HF`                | `(1, 600)` |
| EXP_01   |  `1.169`         | Baseline experiment, `6` Blocks of Transformer Encoder, number of events is restricted to `100` , But I am doing pooling based on `mask` | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_HF_V1` | `(1, 600)` |
| EXP_02   | `1.144`        | Baseline experiment, `6` Blocks of Transformer Encoder, number of events is restricted to `100` , pooling on `mask` with `logLosh`  loss | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_HF_V2`         | `(1, 600)` |
| EXP_03 |`nan` |Baseline experiment, `6` Blocks of Transformer Encoder, number of events is restricted to `100` , pooling on `mask` with `logLosh`  loss, in addition, no `log10` normalization for the charge. `sensor_id` now have there own learnable embeddings with `dim=128`,  `x`, `y` and `z` are normalized between `0` and `1`, time also normalized between `1` and `0`, added `weighted` feature based on time (total `event` features are now `14`). ref: https://www.kaggle.com/code/roberthatch/lb-1-183-lightning-fast-baseline-with-polars| `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_EMBED_V0` | `(1, 600)` |
| EXP_04 | `1.216` -> `nan`  |Baseline experiment, `6` Blocks of Transformer Encoder, number of events is restricted to `100` , pooling on `mask` with `logLosh`  loss, in addition, no `log10` normalization for the charge. `sensor_id` now have there own learnable embeddings with `dim=128`,  `x`, `y` and `z` are normalized between `0` and `1`, time also normalized between `1` and `0` | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_EMBED_V1` | `(1, 600)` |
|EXP_05 |`1.140`| Baseline experiment, `6` Blocks of Transformer Encoder, number of events is restricted to `100` , pooling on `mask` with `logLosh`  loss, in addition, `log10` normalization for the charge. `sensor_id` now have there own learnable embeddings with `dim=128`,  `x`, `y` and `z` are normalized between `0` and `1`, time also normalized between `1` and `0`. This time i did not add `padding_index == 0` to `nn.Embedding` and also removed `post_norma(embed)` normalization of `embeddings`| `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_EMBED_V2` | `(1, 600)` |
|EXP_06|`1.286` | In this experiment, I am trying to use graph `Transformer`, which takes in to account `adjacent matrix` and `distance_matrix`. `adjacent matrix` is calculated by taking `sensor_id` which are `0.015` away from each other (`note`: this might needs to be tuned). `log10` normalization for the charge, `time` is normalized between `1` and `0`, `x`, `y` and `z` are normalized between `0` and `1`. As per usual i restricted to `100` rows per `event`. `6` blocks of `encoders` with `dim=128` and out `2`. note: `Lg=0.5` - weight of adjacent matrix, `Ld=0.5` - weight of distance matrix. Need to optmized, Pooling right now is not done on `mask` but on `x = x.mean(dim=-2)` -> needs to be optmized. Either by testing on `mask` |`!CUDA_VISIBLE_DEVICES=0 python train.py --config_name MATGRAPH`|`(1, 100)` |
|EXP_07 |`1.275`| same as `EXP_06` but `pooling` is done now on `mask` using `MeanPoolingWithMask`  |`!CUDA_VISIBLE_DEVICES=0 python train.py --config_name MATGRAPHV2`|`(1, 100)` |
|EXP_08 |`~ 1.275`| same as `EXP_06` but `pooling` is done now on `mask` using `MeanPoolingWithMask` , `x`, `y` and `z` are normalized by dividing by `500`,  `time` is normalized  `(event['time'] - 1.0e04) / 3.0e4` and `charge` `np.log10(event["charge"])/3.0`  and  `adjacent matrix` is calculated by taking `sensor_id` which are `0.05` away from each other |`!CUDA_VISIBLE_DEVICES=0 python train.py --config_name MATGRAPHV3`|`(1, 100)` |
|EXP_09 |`1.177`| same as `EXP_05`  but with `(1, 100)`  just to have benchmark for small data training  |`!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_EMBED_V3`|`(1, 100)` |
|EXP_10 |`NG`| same as `EXP_09`  but with `SigmoidRange`  |`!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_EMBED_V4`|`(1, 100)` |
|EXP_11 |`SAME`| same as `EXP_09` but extended `max_events` to `160` from `100` |`!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_EMBED_V5` |`(1, 100)` |
|EXP_12 |`1.170`| same as `EXP_02` , but event are restricted to `128`, they are selected based on `pulse` and `light_speed`, pooling is performed using `mask` | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_HF_V3` |`(1, 100)` |
|EXP_13 |`1.143`| This experiment I am using again transformer `encoder` with `6` layers, pooling on `mask`, normalization is performed in following way. For `xyz` we divide by `500` for charge its `log10` and for time its  ` (event["time"].values - 1.0e04) / 3.0e4`, added additional features; `qe` outer layer of the icecube and added `ice_scattering`. The dataset is filtered using `light speed` travel distance if it exceed more then `128` rows| `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_HF_V4` |`(1, 100)` |
|EXP_14 |`1.143`| same as `EXP_13` but with `mean` and `max` masked `pool` concataneted| `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_HF_V5` |`(1, 100)` |


```python
#embeding dimension changed from 150 to 196 for transformer ecnoder
```

https://www.kaggle.com/code/solverworld/icecube-neutrino-path-least-squares-1-214

https://github.com/lucidrains/En-transformer
https://github.com/lucidrains/adjacent-attention-network
https://github.com/lucidrains/equiformer-pytorch
https://github.com/lucidrains/egnn-pytorch
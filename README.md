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
|EXP_14 |`1.142`| same as `EXP_13` but with `mean` and `max` masked `pool` concataneted| `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_HF_V5` |`(1, 100)` |
|EXP_16 |`1.043`| This experiment I am using again transformer encoder with `6` layers, pooling on mask, normalization is performed in following way. For `xyz` we divide by `500` for charge its `log10` and for time its `(event["time"].values - 1.0e04) / 3.0e4`, added additional features; `qe` outer layer of the icecube and added ice_scattering. The dataset is filtered using light speed travel distance if it exceed more then `128` rows, double pooling (`mean` and `max`) using `mask`, loss function  `VonMisesFisher3DLoss`| `!CUDA_VISIBLE_DEVICES=1 python train.py --config_name BASELINE_HF_V7` |`(1, 100)` |
|EXP_17 |`1.017`| same as `EXP_16` but finetuning using weights from `EXP_16` and increased `max_event=148` | `!CUDA_VISIBLE_DEVICES=1 python train.py --config_name BASELINE_HF_V8` |`(1, 100)`|
|EXP_18 |`1.008`/LB: `1.006`| same as `EXP_17` but finetuning using weights from `EXP_17` and with full dataset| `!CUDA_VISIBLE_DEVICES=1 python train.py --config_name BASELINE_HF_V8FT` |`(1, 600)`
|EXP_19 |`~1.043`| same as `EXP_16` but finetuning using weights from `EXP_16` added `absorption` as feature, `max_events==148`, in total `9` features, models now `pools` on `mean`, `max`, `min`, based on the mask, added `ae` like layer in between the pooling | `!CUDA_VISIBLE_DEVICES=1 python train.py --config_name BASELINE_HF_V9` |`(1, 100)` |
|EXP_20|`1.024`| same as `EXP_16` but using `VonMisesFisher3DLoss` and `CosineSimilarityLoss`| `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_HF_V10` |`(1, 100)`|
|EXP_21|`1.025`| got NaN at some point, but loss was still better same as `EXP_16` but using `VonMisesFisher3DLoss` and `EucLadianDistanceLoss`| `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_HF_V11` |`(1, 100)`|
|EXP_22|`ES`| same as `EXP_16` but using `VonMisesFisher3DLoss` and `EucLadianDistanceLoss` and `CosineSimilarityLoss`| `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_HF_V11` |`(1, 100)`|
|EXP_23|`1.02`| same as `EXP_16` but using `VonMisesFisher3DLoss` and `GraphNet`, the `KNN` grouping is performed using `xyzt` and max_events are `196`| `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_graph_V0` |`(1, 100)`|
|EXP_24|`1.016`| same as `EXP_16` but using `VonMisesFisher3DLoss` and `CosineSimilarityLoss`, transformer `encoder` `8` layers, added `rotatry_emb` and `ff_glu`, and `post_emb_normalazation`| `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_HF_V13` |`(1, 100)`|
|EXP_24_CLS|`1.023`| same as `EXP_24` but using `VonMisesFisher3DLoss` and doing pooling `cls` token, the results is slightly worse, but this is expected for small trnasformer| `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_HF_V13_CLS` |`cycle`|
|EXP_24_FT|`1.005`| same as `EXP_24` but using `VonMisesFisher3DLoss` and `CosineSimilarityLoss`, and `FT` at `fp32` due to `NaN`s | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_HF_V14` |`cycle`|
|EXP_24_FT_2|`1.0014`| same as `EXP_24` but using `VonMisesFisher3DLoss` and `CosineSimilarityLoss`. `EuclidDistance`, and `FT` at `fp32` due the lenth of the sequnce is `148` | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_HF_V14_FT` |`cycle`|
|EXP_25|`1.017`| same as `EXP_16` but using `VonMisesFisher3DLoss` and `CosineSimilarityLoss` and `GraphNet`, the `KNN` grouping is performed using `xyz` and max_events are `196`, the first grouping in dataloder is done using `xyzt`, i am cycling thru training (going thru all batches)| `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_graph_V1` |`(1, 100)`|
|EXP_25_FT|`0.999`| same as `EXP_25` but finetuning| `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_graph_V1_FT` |`(1, 100)`|
|EXP_26|`1.037`| same as `EXP_16` but using `VonMisesFisher3DLoss` and `CosineSimilarity` and `EGNNmodel`, the `KNN` grouping is performed using `xyzt` with `8` neighbors,  and max_events are `196`, i am using `5` layer, aggregation type `sum`, embedding `dim = 128` | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_graph_V2` |`cycle`|
|EXP_27|`1.535`| same as `EXP_16` but using `VonMisesFisher3DLoss` and `CosineSimilarity` and `EGNNmodelV1`, the `KNN` grouping is performed using `xyzt` with `8` neighbors,  and max_events are `196`, i am using `5` layer, aggregation type `sum`, embedding `dim = 128`, in `EXP_25` i feed coodinates as embedding, now i will only feed `6` features and keep coordinate seperated, score is really bad | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_graph_V3` |`cycle`|
|EXP_28|`1.050`| same as `EXP_26` but usng `mean`, `max`, `sum`, `min` pooling scheme | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_graph_V4` |`cycle`|
|EXP_29|`1.019`| same as `EXP_05` (`sensor_id`, has there own embeddings), feature input size is `9`,  but using `masked` `mean` and `max` pooliing, transformer encoder `8` layers, with `rotatry_emb` and `ff_glu`, and `post_emb_normalazation`, `attn` dim `256` and loss_func `VonMisesFisher3DLoss`  and `VonMisesFisher3DLossEcludeLossCosine` and `fp32` |`!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_HF_V15` |`cycle`|
|EXP_29_FT|`0.99999`| same as `EXP_29` but with max_len `160` and `fp32`| `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_HF_V15_FT` |`cycle`|
|EXP_29_FT_2|`1.002`| same as `EXP_29_FT` but with max_len `196` and `fp32`| `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_HF_V15_FT_2` |`cycle`|
|EXP_30|`1.0379`| same as `EXP_26`,  added hemophilty as input features| `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_graph_V5` |`cycle`|
|EXP_31|| same as `EXP_30`,  added first layer of `graphnet` as embeding layer and then standarted `EGNNModel`, after first layer we will have `279` embedding features that will go to ENGG along with positions.| `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_graph_V6` |`cycle`|
|EXP_32|`BAD`| same as `EXP_16` but using `VonMisesFisher3DLoss` and `CosineSimilarity` and `EGNNmodelV7`, the `KNN` grouping is performed using `xyzt` with `9` neighbors,  and max_events are `196`, i am using `5` layer, aggregation type `sum`, with `swish` activation function embedding `dim = 128` also i have embeded `sensor_id` with `dim` = `32`, in `EXP_25` i feed coodinates as embedding, now i will only feed `6` features and keep coordinate seperated, i tried something similar in `EXP_27` but without `senosr_id` the score was bad | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_graph_V7` |`cycle`|
|EXP_33|`1.010`| Here I am using embeding layer from `garpnet`, `graphnet` module calculates first feature based on hemophility and then concat them and passes thru `garph` convolution. This is my embedding layer. After this i just feed to standart transfomer with `6` ecnoder `8` heads. pooling is performed on `mask` with `mean` and `max` concatenated, `ff_glue` , and `rotary_pos_emb` , `max_len = 128` | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_graph_V8` |`cycle`|
|EXP_33_FT|`1.0007`| same as EXP_33 but finetuning,  `max_len = 196` | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_graph_V8_FT` |`cycle`|
|EXP_33_FT_2|`NE`| same as EXP_33 but finetuning,  `max_len = 196` | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_graph_V8_FT_2` |`cycle`|
|EXP_33_FT_3_KAPPA|`NE`| same as EXP_33 but finetuning,  `max_len = 196` and filtering based on `kappa > 0.5` | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_graph_V8_FT_3` |`cycle`|
|EXP_34|`1.025`| same as `EXP_26` but i modified `EGNNModel`, every forward pass thru convolution we will try to use `KNN` to rearange edges based on `xyz` (very similar what `dynnet` does | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_graph_V9` |`cycle`|
|EXP_34_FT|`1.010`| same as `EXP_34` but FT using `gVonMisesFisher3DLossCosineSimularityLoss` | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_graph_V9` |`cycle`|
|EXP_34_FT_2|`1.005`| same as `EXP_34_FT_2` but FT | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_graph_V9_FT_2` |`cycle`|
|EXP_35|`same`| same as `EXP_34` but i modified `EGNNModel`, every forward pass thru convolution we will try to use `KNN` to rearange edges based on `pos`,  `xyz` (very similar what `dynnet` does added `two` more features | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_graph_V10` |`cycle`|
|EXP_36|`same`| same us `EXP_33` but added `2` center of gravity feaatures | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_graph_V11` |`cycle`|
|EXP_37|| training transformer based on `iofass` features, added few additiaonl features like a rank, `HuggingFaceDatasetV14`,  `Encoder` is slitly bigger, with `dim_out=256`, `attn_depth = 12`, `heads = 12`, `ff_glu = True`,`rotary_pos_emb = True`,  `use_rmsnorm = True,`, `layer_dropout = 0.1` ,`attn_dropout = 0.1`, `ff_dropout = 0.1`, added `3` pooling, `max`, `mean` and `cls_token`| `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name NANO_TRANSFORMER` |`cycle`|
|EXP_38||training `EGNN` with `10` layers, `GELU` activation function, added features based on `gExtractorV1`, `aux-emb`, `qe - emb`, edge rebuilding on updated `position` based on `7` neighbors, using `xyztc` | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_graph_V12` |`cycle`|
|EXP_39|`1.000009`|training `DynNet` similar to `EXP_29_FT` but with `5` layers, `GELU` activation function using `xyztc` training crashed due to `cpu`| `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_graph_V13` |`cycle`|
|EXP_39_FT|`1.000009`|FT `EXP_39_FT` no improvent loss fluctuates...| `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_graph_V13_FT` |`cycle`|
|EXP_39_FT_2|`1.000009`|FT `EXP_39_FT_2` no improvent loss fluctuates...| `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_graph_V13_FT_2` |`cycle`|
|EXP_40||combining `EXP_25` with Transformer , GNN->Transformer, added residual connection, concat features after GNN and also after Pool, with transformer| `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_graph_V13_FT_2` |`cycle`|
| init5_hb_ft-2.ipynb  | `0.9854` | `EncoderWithDirectionReconstructionV8`, uses ralitive possition baise with scaling,  `dim_out=256`, `attn_depth` = `8`, `heads` = `12` , `layer_dropout = 0.01`, `attn_dropout = 0.01`, ` ff_dropout = 0.01`, epoch `6` | `init5_hb_ft-2` | `full` |
|EXP_100|`0.995`|combining `DynNet` with Transformer , GNN->transformers, I am taking orignal GNN that has cv of `0.99` (its 4 layers) freezing it and using it as feature extractor and feeding to to transformer. Transformer has `cls_token_pooling`, `6` encoder layers. `12` gradd accumulation |`!CUDA_VISIBLE_DEVICES=0 python train.py --config_name EXP_100` |`cycle`|
|EXP_101|`0.994`|similar to `EXP_100` but I am encoding `charge`, `qe`, `aux`, `ice_properties`, concat with original `xyzt` and feeding to 1 layer graphnet (i am not encoding `xyzt`), because i need them for building edges, after this everything get fed to `10` layer transformer `encoder` , I also modified `max_events` we give priority to `aux` and then sort by `time`, `12` gradd accumulation |`!CUDA_VISIBLE_DEVICES=0 python train.py --config_name EXP_101` |`cycle`|
|hb_egnn|`0.988`|since `GraphNET` -> `Transforemr` did not show good perforamnce, I decided to replace `Garphnet` -> `EGGNN` (`Equivariant GNN`), I am taking first `2` layers , `edges` are build based on `xyz` and attaching `6` layers of transformer `encoder` |`hb_training_loop/hb_egnn.ipynb` |`cycle`|
|hb_localattention|`0.983251`|4 local attention, we attend based on `adjacen_matrix` max `8` neighbors, 3 layers + nomal transformer encoder, dropouts only added to local attentions | `hb_training_loop/hb_localattention.ipynb` |`cycle`|
|hb_localattention_ft|`0.9816`|same as `hb_localattention` just finetuning, with added two augmenations, randomly drop 5% events (p < `0.1`) and add randomly up +/- `5ns` (p < `0.1`)| `hb_training_loop/hb_localattention_ft.ipynb` |`cycle`|
|hb_mat||returning to the experiments `EXP_06`, using 4 layers for molecular transfomer but we are only considering `adjnacent matrix` and the weights to be combined with global attention to `0.9`, adjacent matrix is build using `12` neighborns and now we concider `xyzt`, based on previos experiment i addinatlly added to `DeepInceMode` droputs `attn_drop=0.1, drop=0.1`| `hb_training_loop/hb_mat.ipynb` |`cycle`|
|hb_graph_encoder|`0.994`|retrying one last time gnn-> transformer, `xyzt` adjacet matrix on `12` neighbros (xyzt) implementaion might have a bug ... | `hb_training_loop/hb_graph_encoder.ipynb` |`cycle`|
|hb_localattentionV2.ipynb|`0.9839`|sub_version of local attention more like global attention but with `fixed` learnable latent , `EncoderWithDirectionReconstructionV14`| `hb_training_loop/hb_localattentionV2.ipynb` |`cycle`|
|hb_localattentionV3.ipynb|`0.9840`|same as `V2` but integrated `cls_token` at the begining and training with two `augs`, `time`, and `event` dropps, `EncoderWithDirectionReconstructionV15`| `hb_training_loop/hb_localattentionV3ipynb` |`cycle`|
|hb_localattentionV4.ipynb|`0.9816`| 3x (`localattention` with `Factorize Attention`) followed by transformer, `EncoderWithDirectionReconstructionV11_V2_LOCAL_GLOBAL`| `hb_training_loop/hb_localattentionV4.ipynb` |`cycle`|
|hb_localattentionV4FT2.ipynb|same as `hb_localattentionV4` but finetuning on full datasset | `hb_training_loop/hb_localattentionV4FT2.ipynb` |`FULL`|
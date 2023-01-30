### EXPERIMENTS

# Baseline 
#make a table

| EXP_NAME | SCORE     | DESCRIPTION                                                                                                                        | SCRIPT                                        |
| :------- | :-------- | :--------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------- |
| EXP_00   | `1.182` | Baseline experiment, `6` Blocks of Transformer Encoder, number of events is restricted to 100                                        | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_HF` `                |
| EXP_01   |  `1.169`         | Baseline experiment, `6` Blocks of Transformer Encoder, number of events is restricted to `100` , But I am doing pooling based on `mask` | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_HF_V1` |
| EXP_02   | `1.144`        | Baseline experiment, `6` Blocks of Transformer Encoder, number of events is restricted to `100` , pooling on `mask` with `logLosh`  loss | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_HF_V2`         |
| EXP_03 |`nan` |Baseline experiment, `6` Blocks of Transformer Encoder, number of events is restricted to `100` , pooling on `mask` with `logLosh`  loss, in addition, no `log10` normalization for the charge. `sensor_id` now have there own learnable embeddings with `dim=128`,  `x`, `y` and `z` are normalized between `0` and `1`, time also normalized between `1` and `0`, added `weighted` feature based on time (total `event` features are now `14`). ref: https://www.kaggle.com/code/roberthatch/lb-1-183-lightning-fast-baseline-with-polars| `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_EMBED_V0` |

| EXP_04 | |Baseline experiment, `6` Blocks of Transformer Encoder, number of events is restricted to `100` , pooling on `mask` with `logLosh`  loss, in addition, no `log10` normalization for the charge. `sensor_id` now have there own learnable embeddings with `dim=128`,  `x`, `y` and `z` are normalized between `0` and `1`, time also normalized between `1` and `0` | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_EMBED_V1` |



https://www.kaggle.com/code/solverworld/icecube-neutrino-path-least-squares-1-214
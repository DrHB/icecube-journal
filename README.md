### EXPERIMENTS

# Baseline 
#make a table

| EXP_NAME | SCORE     | DESCRIPTION                                                                                                                        | SCRIPT                                        |
| :------- | :-------- | :--------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------- |
| EXP_00   | 1.182 | Baseline experiment, `6` Blocks of Transformer Encoder, number of events is restricted to 100                                        | ``!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_HF` `                |
| EXP_00   |           | Baseline experiment, `6` Blocks of Transformer Encoder, number of events is restricted to `100` , But I am doing pooling based on mask | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_HF_V1` |
| EXP_01   |           | Baseline experiment, `6` Blocks of Transformer Encoder, number of events is restricted to `100` , with `logLosh`    | `!CUDA_VISIBLE_DEVICES=0 python train.py --config_name BASELINE_HF_V2`         |



https://www.kaggle.com/code/solverworld/icecube-neutrino-path-least-squares-1-214
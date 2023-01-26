### EXPERIMENTS

# Baseline 
#make a table

| EXP_NAME | SCORE     | DESCRIPTION                                                                                                                        | SCRIPT                                        |
| :------- | :-------- | :--------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------- |
| EXP_00   | 1.1862279 | Baseline experiment, 3 Blocks of Transformer Encoder, number of events is restricted to 100                                        | icecube/EDA/training_eda.ipynb                |
| EXP_00   |           | Baseline experiment, 3 Blocks of Transformer Encoder, number of events is restricted to 100 , But I am doing pooling based on mask | python train.py --config_name BASELINE_CONFIG |
| EXP_01   |           | Baseline experiment, 3 Blocks of Transformer Encoder, number of events is restricted to 50 , and I am taking only True events      | python train.py --config_name EXP_01          |
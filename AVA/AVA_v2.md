# Information
OCR model based on these archival sources:
* ÖStA/AVA/UuK/NK/IsrK/0029 (https://www.archivinformationssystem.at/detail.aspx?ID=167442)

# Commands
```
ketos compile --workers 3 --random-split 0.8 0.1 0.1 -f page -o ava_v1.arrow *.xml
ketos train --workers 3 --output ava -f binary ava_v1.arrow
```

# Training
```
ketos train -d cuda:0 -i german_handwriting.mlmodel --resize add --workers 8 --output ava_v2 -f binary ava.arrow
scikit-learn version 1.2.2 is not supported. Minimum required version: 0.17. Maximum required version: 1.1.2. Disabling scikit-learn conversion API.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
`Trainer(val_check_interval=1.0)` was configured so validation will run at the end of the training epoch..
You are using a CUDA device ('NVIDIA GeForce RTX 3060') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃    ┃ Name      ┃ Type                     ┃ Params ┃                 In sizes ┃                Out sizes ┃
┡━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 0  │ val_cer   │ CharErrorRate            │      0 │                        ? │                        ? │
│ 1  │ val_wer   │ WordErrorRate            │      0 │                        ? │                        ? │
│ 2  │ net       │ MultiParamSequential     │  4.1 M │  [[1, 1, 120, 400], '?'] │   [[1, 306, 1, 50], '?'] │
│ 3  │ net.C_0   │ ActConv2D                │  1.3 K │  [[1, 1, 120, 400], '?'] │ [[1, 32, 120, 400], '?'] │
│ 4  │ net.Do_1  │ Dropout                  │      0 │ [[1, 32, 120, 400], '?'] │ [[1, 32, 120, 400], '?'] │
│ 5  │ net.Mp_2  │ MaxPool                  │      0 │ [[1, 32, 120, 400], '?'] │  [[1, 32, 60, 200], '?'] │
│ 6  │ net.C_3   │ ActConv2D                │ 40.0 K │  [[1, 32, 60, 200], '?'] │  [[1, 32, 60, 200], '?'] │
│ 7  │ net.Do_4  │ Dropout                  │      0 │  [[1, 32, 60, 200], '?'] │  [[1, 32, 60, 200], '?'] │
│ 8  │ net.Mp_5  │ MaxPool                  │      0 │  [[1, 32, 60, 200], '?'] │  [[1, 32, 30, 100], '?'] │
│ 9  │ net.C_6   │ ActConv2D                │ 55.4 K │  [[1, 32, 30, 100], '?'] │  [[1, 64, 30, 100], '?'] │
│ 10 │ net.Do_7  │ Dropout                  │      0 │  [[1, 64, 30, 100], '?'] │  [[1, 64, 30, 100], '?'] │
│ 11 │ net.Mp_8  │ MaxPool                  │      0 │  [[1, 64, 30, 100], '?'] │   [[1, 64, 15, 50], '?'] │
│ 12 │ net.C_9   │ ActConv2D                │  110 K │   [[1, 64, 15, 50], '?'] │   [[1, 64, 15, 50], '?'] │
│ 13 │ net.Do_10 │ Dropout                  │      0 │   [[1, 64, 15, 50], '?'] │   [[1, 64, 15, 50], '?'] │
│ 14 │ net.S_11  │ Reshape                  │      0 │   [[1, 64, 15, 50], '?'] │   [[1, 960, 1, 50], '?'] │
│ 15 │ net.L_12  │ TransposedSummarizingRNN │  1.9 M │   [[1, 960, 1, 50], '?'] │   [[1, 400, 1, 50], '?'] │
│ 16 │ net.Do_13 │ Dropout                  │      0 │   [[1, 400, 1, 50], '?'] │   [[1, 400, 1, 50], '?'] │
│ 17 │ net.L_14  │ TransposedSummarizingRNN │  963 K │   [[1, 400, 1, 50], '?'] │   [[1, 400, 1, 50], '?'] │
│ 18 │ net.Do_15 │ Dropout                  │      0 │   [[1, 400, 1, 50], '?'] │   [[1, 400, 1, 50], '?'] │
│ 19 │ net.L_16  │ TransposedSummarizingRNN │  963 K │   [[1, 400, 1, 50], '?'] │   [[1, 400, 1, 50], '?'] │
│ 20 │ net.Do_17 │ Dropout                  │      0 │   [[1, 400, 1, 50], '?'] │   [[1, 400, 1, 50], '?'] │
│ 21 │ net.O_18  │ LinSoftmax               │  122 K │   [[1, 400, 1, 50], '?'] │   [[1, 306, 1, 50], '?'] │
└────┴───────────┴──────────────────────────┴────────┴──────────────────────────┴──────────────────────────┘
Trainable params: 4.1 M                                                                                                                                                          
Non-trainable params: 0                                                                                                                                                          
Total params: 4.1 M                                                                                                                                                              
Total estimated model params size (MB): 16                                                                                                                                       
stage 0/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:19 • 0:00:00 16.37it/s val_accuracy: 0.801 val_word_accuracy: 0.424  early_stopping: 0/10 0.80063
stage 1/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:18 • 0:00:00 15.97it/s val_accuracy: 0.825 val_word_accuracy: 0.479  early_stopping: 0/10 0.82485
stage 2/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:16 • 0:00:00 16.16it/s val_accuracy: 0.828 val_word_accuracy: 0.482  early_stopping: 0/10 0.82816
stage 3/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:14 • 0:00:00 16.34it/s val_accuracy: 0.807 val_word_accuracy: 0.453  early_stopping: 1/10 0.82816
stage 4/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:18 • 0:00:00 15.97it/s val_accuracy: 0.0 val_word_accuracy: 0.0  early_stopping: 2/10 0.82816
stage 5/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:11 • 0:00:00 15.34it/s val_accuracy: 0.0 val_word_accuracy: 0.0  early_stopping: 3/10 0.82816
stage 6/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:13 • 0:00:00 16.23it/s val_accuracy: 0.0 val_word_accuracy: 0.0  early_stopping: 4/10 0.82816
stage 7/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:15 • 0:00:00 15.42it/s val_accuracy: 0.0 val_word_accuracy: 0.0  early_stopping: 5/10 0.82816
stage 8/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:32 • 0:00:00 15.18it/s val_accuracy: 0.0 val_word_accuracy: 0.0  early_stopping: 6/10 0.82816
stage 9/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:37 • 0:00:00 14.19it/s val_accuracy: 0.0 val_word_accuracy: 0.0  early_stopping: 7/10 0.82816
stage 10/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:35 • 0:00:00 15.34it/s val_accuracy: 0.001 val_word_accuracy: 0.0  early_stopping: 8/10 0.82816
stage 11/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:21 • 0:00:00 16.29it/s val_accuracy: 0.003 val_word_accuracy: 0.0  early_stopping: 9/10 0.82816
stage 12/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:07 • 0:00:00 14.84it/s val_accuracy: 0.0 val_word_accuracy: 0.0  early_stopping: 10/10 0.82816
Moving best model ava_v2_2.mlmodel (0.8281568288803101) to ava_v2_best.mlmodel
```

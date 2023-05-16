# Information
OCR model based on these archival sources:
* ÖStA/AVA/UuK/NK/IsrK/0029 (https://www.archivinformationssystem.at/detail.aspx?ID=167442)

# Commands
```
ketos compile --workers 3 --random-split 0.8 0.1 0.1 -f page -o ava_v1.arrow *.xml
ketos train -d cuda:0 --workers 3 --output ava -f binary ava_v1.arrow
```

# Training
```
ketos train -d cuda:0 --workers 3 --output ava -f binary ava.arrow
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
│ 2  │ net       │ MultiParamSequential     │  4.0 M │  [[1, 1, 120, 400], '?'] │   [[1, 118, 1, 50], '?'] │
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
│ 21 │ net.O_18  │ LinSoftmax               │ 47.3 K │   [[1, 400, 1, 50], '?'] │   [[1, 118, 1, 50], '?'] │
└────┴───────────┴──────────────────────────┴────────┴──────────────────────────┴──────────────────────────┘
Trainable params: 4.0 M                                                                                                                                                          
Non-trainable params: 0                                                                                                                                                          
Total params: 4.0 M                                                                                                                                                              
Total estimated model params size (MB): 16                                                                                                                                       
stage 0/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:14 • 0:00:00 16.54it/s val_accuracy: 0.08 val_word_accuracy: 0.0  early_stopping: 0/10 0.08005
stage 1/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:06 • 0:00:00 16.21it/s val_accuracy: 0.023 val_word_accuracy: 0.0  early_stopping: 1/10 0.08005
stage 2/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:25 • 0:00:00 15.67it/s val_accuracy: 0.086 val_word_accuracy: 0.001  early_stopping: 0/10 0.08594
stage 3/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:10 • 0:00:00 15.68it/s val_accuracy: 0.149 val_word_accuracy: 0.001  early_stopping: 0/10 0.14887
stage 4/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:09 • 0:00:00 16.32it/s val_accuracy: 0.039 val_word_accuracy: 0.001  early_stopping: 1/10 0.14887
stage 5/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:17 • 0:00:00 16.45it/s val_accuracy: 0.081 val_word_accuracy: 0.002  early_stopping: 2/10 0.14887
stage 6/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:12 • 0:00:00 16.15it/s val_accuracy: 0.089 val_word_accuracy: 0.0  early_stopping: 3/10 0.14887
stage 7/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:11 • 0:00:00 16.28it/s val_accuracy: 0.133 val_word_accuracy: 0.0  early_stopping: 4/10 0.14887
stage 8/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:09 • 0:00:00 16.10it/s val_accuracy: 0.248 val_word_accuracy: -0.418  early_stopping: 0/10 0.24754
stage 9/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:16 • 0:00:00 16.33it/s val_accuracy: 0.245 val_word_accuracy: -0.578  early_stopping: 1/10 0.24754
stage 10/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:15 • 0:00:00 15.92it/s val_accuracy: 0.249 val_word_accuracy: -0.631  early_stopping: 0/10 0.24935
stage 11/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:13 • 0:00:00 16.40it/s val_accuracy: 0.204 val_word_accuracy: -0.093  early_stopping: 1/10 0.24935
stage 12/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:20 • 0:00:00 15.88it/s val_accuracy: 0.24 val_word_accuracy: -0.306  early_stopping: 2/10 0.24935
stage 13/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:36 • 0:00:00 14.92it/s val_accuracy: 0.243 val_word_accuracy: -0.518  early_stopping: 3/10 0.24935
stage 14/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:32 • 0:00:00 15.53it/s val_accuracy: 0.193 val_word_accuracy: -0.089  early_stopping: 4/10 0.24935
stage 15/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:26 • 0:00:00 15.55it/s val_accuracy: 0.147 val_word_accuracy: 0.001  early_stopping: 5/10 0.24935
stage 16/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:29 • 0:00:00 15.54it/s val_accuracy: 0.053 val_word_accuracy: 0.002  early_stopping: 6/10 0.24935
stage 17/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:27 • 0:00:00 15.81it/s val_accuracy: 0.239 val_word_accuracy: -0.481  early_stopping: 7/10 0.24935
stage 18/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:25 • 0:00:00 15.84it/s val_accuracy: 0.151 val_word_accuracy: 0.002  early_stopping: 8/10 0.24935
stage 19/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:25 • 0:00:00 15.74it/s val_accuracy: 0.156 val_word_accuracy: 0.001  early_stopping: 9/10 0.24935
stage 20/∞ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7871/7871 0:08:25 • 0:00:00 15.65it/s val_accuracy: 0.143 val_word_accuracy: 0.002  early_stopping: 10/10 0.24935
Moving best model ava_10.mlmodel (0.24934518337249756) to ava_best.mlmodel
```

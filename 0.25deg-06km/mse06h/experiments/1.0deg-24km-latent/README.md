

## 1x16 encoder

Some of the aspects of the first configuration that ran successfully were a bit
random, since I was changing the config repeatedly while debugging a segmentation
fault issue.
These random aspects were:
* `model.encoder.graph_attention_backend = "pyg"` not "triton", same for
  decoder. Unsure of the impact of this one.
* `dataloader.prefetch_factor = 1`, this one's probably a good idea
* `dataloader.num_workers.training = 2`, with prefetch factor = 1, this could be
  4 or maybe even 8 to speed things up
* `training.multistep_input = 1`, this is fine


```
┏━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name    ┃ Type                 ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ model   │ AnemoiModelInterface │  137 M │ train │     0 │
│ 1 │ loss    │ MSELoss              │      0 │ train │     0 │
│ 2 │ metrics │ ModuleDict           │      0 │ train │     0 │
└───┴─────────┴──────────────────────┴────────┴───────┴───────┘
Trainable params: 137 M
Non-trainable params: 0
Total params: 137 M
Total estimated model params size (MB): 551
Modules in train mode: 281
Modules in eval mode: 0
Total FLOPs: 0
```

```
Epoch 41/-2 ━━━━━━━━━━━━━━━━ 234/234 0:13:47 •        0.55it/s v_num: 0e14
                                     0:00:00                   train_mse_loss_s…
                                                               0.015
                                                               val_mse_loss_ste…
                                                               0.023
                                                               val_mse_loss_epo…
                                                               0.022
                                                               train_mse_loss_e…
                                                               0.015
```

## 2x16 encoder

TODO: Need to git stash pop and debug


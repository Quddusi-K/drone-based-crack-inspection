**U-Net experiments (C–F + ablations), test split, mean per-image metrics unless stated.** `morphology=none` is the raw network output (Experiments C–E); `refined` adds morphological post-processing (Experiment F). Threshold = best of the sweep.

| experiment             | preprocessing   | augmentation   | loss     | morphology   |   threshold |   iou |   dice |   precision |   recall |    f1 |   crack-only Dice |   pooled IoU |   pooled Dice |   ms/img |
|:-----------------------|:----------------|:---------------|:---------|:-------------|------------:|------:|-------:|------------:|---------:|------:|------------------:|-------------:|--------------:|---------:|
| unet_raw               | raw             | off            | bce_dice | none         |         0.5 | 0.437 |  0.518 |       0.471 |    0.602 | 0.518 |             0.578 |        0.457 |         0.627 |     50.1 |
| unet_raw               | raw             | off            | bce_dice | refined      |         0.5 | 0.586 |  0.667 |       0.619 |    0.754 | 0.667 |             0.576 |        0.459 |         0.629 |     50.1 |
| unet_clahe             | contrast        | off            | bce_dice | none         |         0.5 | 0.391 |  0.472 |       0.424 |    0.559 | 0.472 |             0.586 |        0.48  |         0.648 |     49.1 |
| unet_clahe             | contrast        | off            | bce_dice | refined      |         0.5 | 0.588 |  0.669 |       0.621 |    0.758 | 0.669 |             0.58  |        0.475 |         0.644 |     49.1 |
| unet_illum             | illumination    | off            | bce_dice | none         |         0.5 | 0.544 |  0.625 |       0.581 |    0.703 | 0.625 |             0.59  |        0.477 |         0.646 |     50   |
| unet_illum             | illumination    | off            | bce_dice | refined      |         0.5 | 0.641 |  0.722 |       0.678 |    0.802 | 0.722 |             0.585 |        0.474 |         0.643 |     50   |
| unet_aug               | raw             | on             | bce_dice | none         |         0.5 | 0.234 |  0.29  |       0.398 |    0.321 | 0.29  |             0.346 |        0.297 |         0.458 |     47.5 |
| unet_aug               | raw             | on             | bce_dice | refined      |         0.5 | 0.533 |  0.587 |       0.571 |    0.619 | 0.587 |             0.34  |        0.3   |         0.461 |     47.5 |
| unet_clahe_aug         | contrast        | on             | bce_dice | none         |         0.5 | 0.534 |  0.616 |       0.568 |    0.703 | 0.616 |             0.484 |        0.371 |         0.542 |     56.5 |
| unet_clahe_aug         | contrast        | on             | bce_dice | refined      |         0.5 | 0.633 |  0.715 |       0.666 |    0.803 | 0.715 |             0.481 |        0.371 |         0.541 |     56.5 |
| unet_clahe_aug_tversky | contrast        | on             | tversky  | none         |         0.8 | 0.288 |  0.367 |       0.407 |    0.355 | 0.367 |             0.486 |        0.225 |         0.367 |     48.6 |
| unet_clahe_aug_tversky | contrast        | on             | tversky  | refined      |         0.8 | 0.332 |  0.409 |       0.453 |    0.396 | 0.409 |             0.471 |        0.225 |         0.367 |     48.6 |
| unet_clahe_aug_focal   | contrast        | on             | focal    | none         |         0.5 | 0.341 |  0.417 |       0.511 |    0.372 | 0.417 |             0.395 |        0.284 |         0.442 |     47.9 |
| unet_clahe_aug_focal   | contrast        | on             | focal    | refined      |         0.5 | 0.593 |  0.665 |       0.766 |    0.625 | 0.665 |             0.391 |        0.298 |         0.459 |     47.9 |
| unet_clahe_aug_bce     | contrast        | on             | bce      | none         |         0.4 | 0.009 |  0.018 |       0.009 |    0.227 | 0.018 |             0.032 |        0.008 |         0.015 |     47.6 |
| unet_clahe_aug_bce     | contrast        | on             | bce      | refined      |         0.4 | 0.008 |  0.015 |       0.008 |    0.499 | 0.015 |             0.027 |        0.008 |         0.015 |     47.6 |

**Best classical methods (A/B) for reference, same metric definitions (evaluated on all 199 samples):**

| experiment                        | preprocessing   | augmentation   | loss   | morphology            | threshold   |   iou |   dice |   precision |   recall |    f1 |   crack-only Dice |   pooled IoU |   pooled Dice |   ms/img |
|:----------------------------------|:----------------|:---------------|:-------|:----------------------|:------------|------:|-------:|------------:|---------:|------:|------------------:|-------------:|--------------:|---------:|
| E14_illum_blackhat_fixed_morph_cc | illumination    | n/a            | n/a    | [['close', 5]]+cc>=80 | n/a         | 0.117 |  0.167 |       0.199 |    0.281 | 0.167 |             0.278 |        0.038 |         0.072 |     32.4 |
| E13_raw_blackhat_fixed_morph_cc   | raw             | n/a            | n/a    | [['close', 5]]+cc>=80 | n/a         | 0.11  |  0.16  |       0.189 |    0.277 | 0.16  |             0.274 |        0.038 |         0.073 |      2.3 |
| E3_illum_canny                    | illumination    | n/a            | n/a    | []                    | n/a         | 0.045 |  0.08  |       0.084 |    0.121 | 0.08  |             0.157 |        0.032 |         0.062 |     30.8 |

![ablation](results/ablation_dice.png)


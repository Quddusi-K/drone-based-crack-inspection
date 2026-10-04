# Evaluation: unet_clahe_aug_focal (test, n=20)

threshold 0.5 (selected on the validation split by pixel-pooled Dice); inference 47.9 ms/image on cpu

| setting | iou | dice | precision | recall | f1 | accuracy | specificity | pos_iou | pos_dice | global_iou | global_dice |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t0.5 / raw | 0.3413 | 0.4173 | 0.5114 | 0.3721 | 0.4173 | 0.9936 | 0.9984 | 0.2569 | 0.3951 | 0.2836 | 0.4419 |
| t0.5 / refined | 0.5930 | 0.6649 | 0.7662 | 0.6248 | 0.6649 | 0.9939 | 0.9986 | 0.2599 | 0.3907 | 0.2980 | 0.4592 |

## Error analysis counts

| category | count |
|---|---|
| false_positive | 0 |
| false_negative | 4 |
| boundary_error | 4 |
| fragmentation | 4 |
| merging | 1 |
| good | 10 |

## Crack measurement errors (pred mask vs GT mask, pixels)

| metric | value |
|---|---|
| crack_length_pixels_mae | 187.8432 |
| crack_length_pixels_rmse | 223.1490 |
| mean_width_pixels_mae | 0.5086 |
| mean_width_pixels_rmse | 0.7473 |
| max_width_pixels_mae | 1.4159 |
| max_width_pixels_rmse | 1.7156 |
| crack_area_pixels_mae | 400.2727 |
| crack_area_pixels_rmse | 502.2070 |
| crack_area_relative_error | 0.4901 |
| orientation_mean_abs_angular_error | 3.2676 |
| n_images | 11 |

# Evaluation: unet_clahe_aug (test, n=20)

threshold 0.5 (selected on the validation split by pixel-pooled Dice); inference 56.5 ms/image on cpu

| setting | iou | dice | precision | recall | f1 | accuracy | specificity | pos_iou | pos_dice | global_iou | global_dice |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t0.5 / raw | 0.5342 | 0.6161 | 0.5683 | 0.7031 | 0.6161 | 0.9913 | 0.9934 | 0.3349 | 0.4839 | 0.3714 | 0.5416 |
| t0.5 / refined | 0.6329 | 0.7146 | 0.6663 | 0.8032 | 0.7146 | 0.9913 | 0.9933 | 0.3325 | 0.4811 | 0.3709 | 0.5411 |

## Error analysis counts

| category | count |
|---|---|
| false_positive | 0 |
| false_negative | 1 |
| boundary_error | 5 |
| fragmentation | 0 |
| merging | 4 |
| good | 11 |

## Crack measurement errors (pred mask vs GT mask, pixels)

| metric | value |
|---|---|
| crack_length_pixels_mae | 111.4014 |
| crack_length_pixels_rmse | 135.7040 |
| mean_width_pixels_mae | 3.2730 |
| mean_width_pixels_rmse | 3.3864 |
| max_width_pixels_mae | 3.5631 |
| max_width_pixels_rmse | 4.0632 |
| crack_area_pixels_mae | 525.1818 |
| crack_area_pixels_rmse | 587.7151 |
| crack_area_relative_error | 0.7474 |
| orientation_mean_abs_angular_error | 2.6267 |
| n_images | 11 |

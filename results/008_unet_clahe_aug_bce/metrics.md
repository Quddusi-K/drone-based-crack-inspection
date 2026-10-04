# Evaluation: unet_clahe_aug_bce (test, n=20)

threshold 0.4 (selected on the validation split by pixel-pooled Dice); inference 47.6 ms/image on cpu

| setting | iou | dice | precision | recall | f1 | accuracy | specificity | pos_iou | pos_dice | global_iou | global_dice |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t0.5 / raw | 0.2502 | 0.2503 | 0.2531 | 0.2502 | 0.2503 | 0.9925 | 0.9998 | 0.0003 | 0.0006 | 0.0002 | 0.0004 |
| t0.5 / refined | 0.4500 | 0.4500 | 0.4500 | 0.4500 | 0.4500 | 0.9927 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| t0.4 / raw | 0.0090 | 0.0175 | 0.0093 | 0.2275 | 0.0175 | 0.6651 | 0.6679 | 0.0163 | 0.0319 | 0.0075 | 0.0149 |
| t0.4 / refined | 0.0076 | 0.0150 | 0.0077 | 0.4993 | 0.0150 | 0.1471 | 0.1416 | 0.0139 | 0.0274 | 0.0076 | 0.0151 |

## Error analysis counts

| category | count |
|---|---|
| false_positive | 9 |
| false_negative | 0 |
| boundary_error | 0 |
| fragmentation | 3 |
| merging | 1 |
| good | 7 |

## Crack measurement errors (pred mask vs GT mask, pixels)

| metric | value |
|---|---|
| crack_length_pixels_mae | 2682.0174 |
| crack_length_pixels_rmse | 3032.7431 |
| mean_width_pixels_mae | 58.2560 |
| mean_width_pixels_rmse | 72.2120 |
| max_width_pixels_mae | 178.1263 |
| max_width_pixels_rmse | 187.4250 |
| crack_area_pixels_mae | 55644.0000 |
| crack_area_pixels_rmse | 55954.6026 |
| crack_area_relative_error | 2036.2935 |
| orientation_mean_abs_angular_error | 56.8362 |
| n_images | 11 |

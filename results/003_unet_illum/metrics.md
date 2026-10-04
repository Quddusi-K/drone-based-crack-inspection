# Evaluation: unet_illum (test, n=20)

threshold 0.5 (selected on the validation split by pixel-pooled Dice); inference 50.0 ms/image on cpu

| setting | iou | dice | precision | recall | f1 | accuracy | specificity | pos_iou | pos_dice | global_iou | global_dice |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t0.5 / raw | 0.5437 | 0.6247 | 0.5815 | 0.7026 | 0.6247 | 0.9937 | 0.9951 | 0.4431 | 0.5903 | 0.4766 | 0.6456 |
| t0.5 / refined | 0.6410 | 0.7218 | 0.6777 | 0.8022 | 0.7218 | 0.9936 | 0.9951 | 0.4381 | 0.5851 | 0.4742 | 0.6433 |

## Error analysis counts

| category | count |
|---|---|
| false_positive | 1 |
| false_negative | 1 |
| boundary_error | 5 |
| fragmentation | 0 |
| merging | 2 |
| good | 12 |

## Crack measurement errors (pred mask vs GT mask, pixels)

| metric | value |
|---|---|
| crack_length_pixels_mae | 82.4001 |
| crack_length_pixels_rmse | 109.6080 |
| mean_width_pixels_mae | 2.0444 |
| mean_width_pixels_rmse | 2.1106 |
| max_width_pixels_mae | 1.9983 |
| max_width_pixels_rmse | 2.1116 |
| crack_area_pixels_mae | 365.1818 |
| crack_area_pixels_rmse | 468.4647 |
| crack_area_relative_error | 0.5949 |
| orientation_mean_abs_angular_error | 1.5669 |
| n_images | 11 |

# Evaluation: unet_raw (test, n=20)

threshold 0.5 (selected on the validation split by pixel-pooled Dice); inference 50.1 ms/image on cpu

| setting | iou | dice | precision | recall | f1 | accuracy | specificity | pos_iou | pos_dice | global_iou | global_dice |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t0.5 / raw | 0.4367 | 0.5177 | 0.4712 | 0.6015 | 0.5177 | 0.9931 | 0.9946 | 0.4303 | 0.5776 | 0.4572 | 0.6275 |
| t0.5 / refined | 0.5860 | 0.6670 | 0.6190 | 0.7540 | 0.6670 | 0.9931 | 0.9946 | 0.4292 | 0.5764 | 0.4587 | 0.6289 |

## Error analysis counts

| category | count |
|---|---|
| false_positive | 2 |
| false_negative | 1 |
| boundary_error | 5 |
| fragmentation | 0 |
| merging | 1 |
| good | 12 |

## Crack measurement errors (pred mask vs GT mask, pixels)

| metric | value |
|---|---|
| crack_length_pixels_mae | 79.2742 |
| crack_length_pixels_rmse | 106.7239 |
| mean_width_pixels_mae | 2.1326 |
| mean_width_pixels_rmse | 2.1828 |
| max_width_pixels_mae | 1.9250 |
| max_width_pixels_rmse | 2.2119 |
| crack_area_pixels_mae | 410.0909 |
| crack_area_pixels_rmse | 491.5019 |
| crack_area_relative_error | 2.1530 |
| orientation_mean_abs_angular_error | 6.2562 |
| n_images | 11 |

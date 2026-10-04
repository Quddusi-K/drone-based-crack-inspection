# Evaluation: unet_aug (test, n=20)

threshold 0.5 (selected on the validation split by pixel-pooled Dice); inference 47.5 ms/image on cpu

| setting | iou | dice | precision | recall | f1 | accuracy | specificity | pos_iou | pos_dice | global_iou | global_dice |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t0.5 / raw | 0.2345 | 0.2901 | 0.3980 | 0.3212 | 0.2901 | 0.9925 | 0.9965 | 0.2445 | 0.3456 | 0.2968 | 0.4577 |
| t0.5 / refined | 0.5326 | 0.5872 | 0.5715 | 0.6189 | 0.5872 | 0.9926 | 0.9967 | 0.2411 | 0.3403 | 0.2996 | 0.4611 |

## Error analysis counts

| category | count |
|---|---|
| false_positive | 1 |
| false_negative | 5 |
| boundary_error | 4 |
| fragmentation | 1 |
| merging | 1 |
| good | 10 |

## Crack measurement errors (pred mask vs GT mask, pixels)

| metric | value |
|---|---|
| crack_length_pixels_mae | 208.1877 |
| crack_length_pixels_rmse | 270.7623 |
| mean_width_pixels_mae | 2.5853 |
| mean_width_pixels_rmse | 2.6351 |
| max_width_pixels_mae | 3.6699 |
| max_width_pixels_rmse | 4.3816 |
| crack_area_pixels_mae | 532.6364 |
| crack_area_pixels_rmse | 661.7842 |
| crack_area_relative_error | 7.4421 |
| orientation_mean_abs_angular_error | 13.0957 |
| n_images | 11 |

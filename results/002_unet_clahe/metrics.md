# Evaluation: unet_clahe (test, n=20)

threshold 0.5 (selected on the validation split by pixel-pooled Dice); inference 49.1 ms/image on cpu

| setting | iou | dice | precision | recall | f1 | accuracy | specificity | pos_iou | pos_dice | global_iou | global_dice |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t0.5 / raw | 0.3909 | 0.4721 | 0.4245 | 0.5589 | 0.4721 | 0.9937 | 0.9950 | 0.4379 | 0.5856 | 0.4797 | 0.6484 |
| t0.5 / refined | 0.5875 | 0.6689 | 0.6210 | 0.7575 | 0.6689 | 0.9936 | 0.9950 | 0.4319 | 0.5799 | 0.4752 | 0.6443 |

## Error analysis counts

| category | count |
|---|---|
| false_positive | 2 |
| false_negative | 1 |
| boundary_error | 5 |
| fragmentation | 1 |
| merging | 3 |
| good | 10 |

## Crack measurement errors (pred mask vs GT mask, pixels)

| metric | value |
|---|---|
| crack_length_pixels_mae | 85.2484 |
| crack_length_pixels_rmse | 113.9551 |
| mean_width_pixels_mae | 2.1035 |
| mean_width_pixels_rmse | 2.1667 |
| max_width_pixels_mae | 1.8903 |
| max_width_pixels_rmse | 2.1230 |
| crack_area_pixels_mae | 414.0909 |
| crack_area_pixels_rmse | 506.2420 |
| crack_area_relative_error | 0.6474 |
| orientation_mean_abs_angular_error | 1.8871 |
| n_images | 11 |

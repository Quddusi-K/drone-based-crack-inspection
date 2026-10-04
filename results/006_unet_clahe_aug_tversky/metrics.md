# Evaluation: unet_clahe_aug_tversky (test, n=20)

threshold 0.8 (selected on the validation split by pixel-pooled Dice); inference 48.6 ms/image on cpu

| setting | iou | dice | precision | recall | f1 | accuracy | specificity | pos_iou | pos_dice | global_iou | global_dice |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t0.5 / raw | 0.1883 | 0.2693 | 0.2057 | 0.4229 | 0.2693 | 0.9569 | 0.9579 | 0.3424 | 0.4896 | 0.1242 | 0.2210 |
| t0.5 / refined | 0.2376 | 0.3186 | 0.2552 | 0.4722 | 0.3186 | 0.9558 | 0.9568 | 0.3412 | 0.4883 | 0.1210 | 0.2159 |
| t0.8 / raw | 0.2876 | 0.3672 | 0.4071 | 0.3547 | 0.3672 | 0.9873 | 0.9908 | 0.3411 | 0.4858 | 0.2251 | 0.3674 |
| t0.8 / refined | 0.3318 | 0.4091 | 0.4531 | 0.3961 | 0.4091 | 0.9876 | 0.9912 | 0.3305 | 0.4710 | 0.2251 | 0.3675 |

## Error analysis counts

| category | count |
|---|---|
| false_positive | 6 |
| false_negative | 2 |
| boundary_error | 2 |
| fragmentation | 2 |
| merging | 1 |
| good | 8 |

## Crack measurement errors (pred mask vs GT mask, pixels)

| metric | value |
|---|---|
| crack_length_pixels_mae | 162.8421 |
| crack_length_pixels_rmse | 201.7047 |
| mean_width_pixels_mae | 1.0810 |
| mean_width_pixels_rmse | 1.2616 |
| max_width_pixels_mae | 1.7438 |
| max_width_pixels_rmse | 2.0737 |
| crack_area_pixels_mae | 322.8182 |
| crack_area_pixels_rmse | 388.4731 |
| crack_area_relative_error | 0.4463 |
| orientation_mean_abs_angular_error | 6.1708 |
| n_images | 11 |

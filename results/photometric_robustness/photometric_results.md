# Robustness to photometric variation (split=test, n=20)

| method | perturbation | iou | dice | precision | recall | f1 |
|---|---|---|---|---|---|---|
| classical:E14_illum_blackhat_fixed_morph_cc | original | 0.1562 | 0.2150 | 0.2065 | 0.3790 | 0.2150 |
| classical:E14_illum_blackhat_fixed_morph_cc | bright+60 | 0.2047 | 0.2634 | 0.2548 | 0.4306 | 0.2634 |
| classical:E14_illum_blackhat_fixed_morph_cc | dark-60 | 0.1555 | 0.2140 | 0.2090 | 0.3718 | 0.2140 |
| classical:E14_illum_blackhat_fixed_morph_cc | low_contrast | 0.2048 | 0.2093 | 0.3306 | 0.2049 | 0.2093 |
| classical:E14_illum_blackhat_fixed_morph_cc | high_contrast | 0.0278 | 0.0465 | 0.0302 | 0.4696 | 0.0465 |
| classical:E14_illum_blackhat_fixed_morph_cc | gamma_0.5 | 0.2386 | 0.2906 | 0.4706 | 0.2610 | 0.2906 |
| classical:E14_illum_blackhat_fixed_morph_cc | gamma_2.2 | 0.1028 | 0.1385 | 0.1750 | 0.3174 | 0.1385 |
| classical:E14_illum_blackhat_fixed_morph_cc | shadow | 0.1329 | 0.1824 | 0.3096 | 0.1683 | 0.1824 |
| classical:E14_illum_blackhat_fixed_morph_cc | local_illumination | 0.1621 | 0.2251 | 0.2538 | 0.3146 | 0.2251 |
| classical:E14_illum_blackhat_fixed_morph_cc | color_shift | 0.1544 | 0.2128 | 0.2054 | 0.3747 | 0.2128 |
| classical:E14_illum_blackhat_fixed_morph_cc | low_light_noise | 0.1322 | 0.1566 | 0.3040 | 0.1350 | 0.1566 |
| unet:002_unet_clahe | original | 0.5398 | 0.6210 | 0.5724 | 0.7100 | 0.6210 |
| unet:002_unet_clahe | bright+60 | 0.4084 | 0.4143 | 0.4342 | 0.4091 | 0.4143 |
| unet:002_unet_clahe | dark-60 | 0.2371 | 0.2808 | 0.3068 | 0.2789 | 0.2808 |
| unet:002_unet_clahe | low_contrast | 0.6788 | 0.7594 | 0.7597 | 0.7844 | 0.7594 |
| unet:002_unet_clahe | high_contrast | 0.4831 | 0.5644 | 0.5014 | 0.6893 | 0.5644 |
| unet:002_unet_clahe | gamma_0.5 | 0.4000 | 0.4000 | 0.4000 | 0.4000 | 0.4000 |
| unet:002_unet_clahe | gamma_2.2 | 0.2432 | 0.2681 | 0.3140 | 0.2547 | 0.2681 |
| unet:002_unet_clahe | shadow | 0.4475 | 0.5266 | 0.5903 | 0.4933 | 0.5266 |
| unet:002_unet_clahe | local_illumination | 0.6761 | 0.7573 | 0.7298 | 0.8035 | 0.7573 |
| unet:002_unet_clahe | color_shift | 0.3753 | 0.4566 | 0.3927 | 0.5906 | 0.4566 |
| unet:002_unet_clahe | low_light_noise | 0.4500 | 0.4500 | 0.4500 | 0.4500 | 0.4500 |

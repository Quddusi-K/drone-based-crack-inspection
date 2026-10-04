# Classical CV comparison (split=all, n=199)

| experiment | preprocessing | method | morphology | iou | dice | precision | recall | f1 | pos_iou | pos_dice | global_dice | inference_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1_raw_canny | raw | classical/canny | [] | 0.0448 | 0.0795 | 0.0818 | 0.1213 | 0.0795 | 0.0873 | 0.1550 | 0.0618 | 0.5610 |
| E2_clahe_canny | contrast | classical/canny | [] | 0.0091 | 0.0176 | 0.0099 | 0.1623 | 0.0176 | 0.0178 | 0.0343 | 0.0191 | 2.0530 |
| E3_illum_canny | illumination | classical/canny | [] | 0.0453 | 0.0803 | 0.0839 | 0.1208 | 0.0803 | 0.0885 | 0.1568 | 0.0617 | 30.8265 |
| E4_clahe_adaptive | contrast | classical/adaptive | [] | 0.0165 | 0.0316 | 0.0168 | 0.3591 | 0.0316 | 0.0322 | 0.0617 | 0.0341 | 1.9029 |
| E5_clahe_canny_morph | contrast | classical/canny | [['close', 5]] | 0.0109 | 0.0205 | 0.0110 | 0.4429 | 0.0205 | 0.0212 | 0.0401 | 0.0207 | 2.0251 |
| E6_combined_adaptive_morph_cc | combined | classical/adaptive | [['open', 3], ['close', 5]]+cc>=80 | 0.0391 | 0.0700 | 0.0409 | 0.3863 | 0.0700 | 0.0764 | 0.1367 | 0.0618 | 44.4757 |
| E7_raw_otsu | raw | classical/otsu | [] | 0.0162 | 0.0309 | 0.0164 | 0.4332 | 0.0309 | 0.0316 | 0.0603 | 0.0372 | 0.2861 |
| E8_raw_sobel | raw | classical/sobel | [] | 0.0414 | 0.0750 | 0.0517 | 0.1668 | 0.0750 | 0.0808 | 0.1463 | 0.0887 | 2.5225 |
| E9_raw_laplacian | raw | classical/laplacian | [] | 0.0315 | 0.0585 | 0.0422 | 0.1187 | 0.0585 | 0.0614 | 0.1142 | 0.0707 | 2.1757 |
| E10_raw_blackhat_otsu | raw | classical/blackhat_otsu | [] | 0.0215 | 0.0389 | 0.0234 | 0.4041 | 0.0389 | 0.0418 | 0.0759 | 0.0418 | 1.0713 |
| E11_clahe_blackhat_morph_cc | contrast | classical/blackhat_otsu | [['close', 5]]+cc>=80 | 0.0117 | 0.0226 | 0.0117 | 0.4923 | 0.0226 | 0.0229 | 0.0441 | 0.0259 | 3.1845 |
| E12_combined_blackhat_morph_cc | combined | classical/blackhat_otsu | [['close', 5]]+cc>=80 | 0.0199 | 0.0360 | 0.0205 | 0.4706 | 0.0360 | 0.0388 | 0.0703 | 0.0354 | 44.7842 |
| E13_raw_blackhat_fixed_morph_cc | raw | classical/blackhat_fixed | [['close', 5]]+cc>=80 | 0.1102 | 0.1605 | 0.1892 | 0.2775 | 0.1605 | 0.1757 | 0.2739 | 0.0730 | 2.2531 |
| E14_illum_blackhat_fixed_morph_cc | illumination | classical/blackhat_fixed | [['close', 5]]+cc>=80 | 0.1167 | 0.1675 | 0.1985 | 0.2811 | 0.1675 | 0.1787 | 0.2778 | 0.0723 | 32.4495 |

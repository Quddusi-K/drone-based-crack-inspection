# Crack/background separability (n=100)

| preset | steps | contrast | fisher_ratio | time_ms |
|---|---|---|---|---|
| raw | [] | 0.0909 | 1.0546 | 0.0019 |
| normalized | ['minmax'] | 0.1429 | 1.0542 | 3.6797 |
| contrast | ['clahe'] | 0.1785 | 1.0536 | 0.5466 |
| illumination | ['background_correction'] | 0.0947 | 1.2179 | 29.1701 |
| retinex | ['retinex'] | 0.1077 | 1.0294 | 12.2337 |
| combined | ['background_correction', {'name': 'clahe', 'clip_limit': 2.0}, {'name': 'bilateral'}] | 0.1762 | 1.3264 | 41.7563 |

## Fisher ratio under photometric perturbation (first 40 images)

| perturbation | raw | contrast | illumination | combined |
|---|---|---|---|---|
| original | 1.1281 | 1.1069 | 1.2344 | 1.3705 |
| bright+60 | 1.1292 | 1.2478 | 1.2284 | 1.5289 |
| dark-60 | 1.1375 | 0.9664 | 1.2476 | 1.2001 |
| low_contrast | 1.1293 | 0.9860 | 1.2210 | 1.4263 |
| high_contrast | 1.1288 | 1.1438 | 1.2391 | 1.3075 |
| gamma_0.5 | 1.0988 | 1.2090 | 1.1883 | 1.5720 |
| gamma_2.2 | 1.1253 | 0.8969 | 1.2535 | 1.1523 |
| shadow | 0.2174 | 0.4842 | 1.1029 | 1.1951 |
| local_illumination | 0.2483 | 0.5980 | 1.2114 | 1.2994 |
| color_shift | 1.1293 | 1.1041 | 1.2378 | 1.3680 |
| low_light_noise | 0.4989 | 0.3768 | 0.5221 | 0.6921 |

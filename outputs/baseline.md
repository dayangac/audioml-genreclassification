# SVM baseline, GTZAN fault-filtered split

train 442 / valid 197 / test 290 clips, 145 summary features

| features | dims | C | gamma | valid acc | test acc |
|---|---|---|---|---|---|
| all | 145 | 10 | 0.001 | 0.635 | 0.624 |
| timbre | 106 | 100 | 0.001 | 0.574 | 0.610 |
| harmony | 36 | 10 | 0.01 | 0.315 | 0.303 |
| rhythm | 3 | 100 | 0.01 | 0.320 | 0.366 |
| timbre + harmony | 142 | 10 | 0.001 | 0.614 | 0.600 |

## Test confusion matrix (all features; rows = true)

| | blue | clas | coun | disc | hiph | jazz | meta | pop | regg | rock |
|---|---|---|---|---|---|---|---|---|---|---|
| blues | 10 | 0 | 0 | 5 | 0 | 8 | 3 | 0 | 2 | 3 |
| classical | 0 | 31 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| country | 4 | 1 | 14 | 6 | 0 | 0 | 1 | 0 | 0 | 4 |
| disco | 1 | 0 | 0 | 17 | 5 | 0 | 0 | 1 | 3 | 2 |
| hiphop | 0 | 0 | 0 | 6 | 17 | 0 | 0 | 3 | 1 | 0 |
| jazz | 0 | 1 | 1 | 0 | 0 | 21 | 0 | 1 | 0 | 3 |
| metal | 0 | 0 | 1 | 1 | 1 | 0 | 23 | 0 | 0 | 1 |
| pop | 0 | 0 | 1 | 2 | 3 | 0 | 0 | 22 | 0 | 2 |
| reggae | 0 | 1 | 1 | 1 | 4 | 0 | 0 | 3 | 16 | 0 |
| rock | 4 | 1 | 0 | 11 | 0 | 1 | 0 | 1 | 4 | 10 |

Per-genre test recall: blues 0.32, classical 1.00, country 0.47, disco 0.59, hiphop 0.63, jazz 0.78, metal 0.85, pop 0.73, reggae 0.62, rock 0.31

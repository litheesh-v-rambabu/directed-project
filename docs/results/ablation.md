Test split (synthetic proxy data), mean ± std over 3 training seeds. Operating point: threshold calibrated on the validation split for 5% false-accept rate (FAR).

| Model | Params | Accuracy % | F1 % | Recall % | FRR % @5% FAR | EER % | AUC | ECE % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Audio only | 308,930 | 82.5 ± 0.6 | 80.5 ± 0.8 | 70.6 ± 1.4 | 29.4 ± 1.4 | 13.6 ± 0.9 | 0.940 ± 0.003 | 5.6 ± 2.0 |
| Text only | 189,794 | 61.8 ± 0.8 | 44.0 ± 2.4 | 29.3 ± 2.3 | 70.7 ± 2.3 | 17.3 ± 0.2 | 0.904 ± 0.001 | 4.0 ± 0.4 |
| Audio + text | 498,435 | 90.8 ± 0.1 | 90.7 ± 0.1 | 87.0 ± 0.4 | 13.0 ± 0.4 | 7.5 ± 0.1 | 0.978 ± 0.001 | 4.8 ± 2.3 |
| Audio + text + speaker | 552,771 | 91.4 ± 0.4 | 91.4 ± 0.4 | 88.7 ± 0.5 | 11.3 ± 0.5 | 7.8 ± 0.4 | 0.978 ± 0.001 | 2.1 ± 0.4 |

Audio + text + speaker model evaluated with the enrollment profile withheld at inference: EER 8.0 ± 0.5 %, FRR @5% FAR 11.6 ± 1.2 %.

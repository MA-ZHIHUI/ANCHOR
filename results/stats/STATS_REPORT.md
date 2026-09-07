# ANCHOR paired statistics (seed=42)

Bootstrap resamples: 10000; EDI permutation: 10000.

## Intervention / defense (Round-4 Acc)

```
                                   setting   n  n11  n00  n01_rescue  n10_harm    mcnemar_p  delta_acc_pp  boot_ci95_lo  boot_ci95_hi  boot_mean  base_acc  interv_acc
 MMLU ID ANCHOR canonical (joint Baseline)  40    2   11          27         0 1.490116e-08     67.500000     52.500000     82.500000  67.541750       5.0   72.500000
     MMLU ID ANCHOR vs frozen REF Baseline  40    2   10          27         1 2.160668e-07     65.000000     47.500000     80.000000  64.985500       7.5   72.500000
     MMLU ID ANCHOR frozen-TRIWIN protocol  40    2   12          25         1 8.046627e-07     60.000000     42.500000     75.000000  59.862250       7.5   67.500000
       MMLU ID system-prompt vs frozen REF  40    2   26          11         1 6.347656e-03     25.000000     10.000000     40.000000  25.035750       7.5   32.500000
MMLU ID single-layer L20 CAA vs frozen REF  40    0   30           7         3 3.437500e-01     10.000000     -5.000000     25.000000  10.028250       7.5   17.500000
                 CSQA OOD ANCHOR canonical 300    7  110         181         2 2.746664e-51     59.666667     54.000000     65.333333  59.641700       3.0   62.666667
CSQA OOD extract-size |E|=30 (exploratory) 300    8  124         167         1 9.033945e-49     55.333333     49.666667     61.000000  55.391200       3.0   58.333333
CSQA OOD extract-size |E|=40 (exploratory) 300    7  110         181         2 2.746664e-51     59.666667     54.000000     65.333333  59.659733       3.0   62.666667
CSQA OOD extract-size |E|=60 (exploratory) 300    7  144         147         2 3.132182e-41     48.333333     42.333333     54.000000  48.281433       3.0   51.333333
CSQA OOD extract-size |E|=80 (exploratory) 300    6  183         108         3 1.756695e-28     35.000000     29.666667     40.666667  35.019267       3.0   38.000000
```

## EDI MT − ST (Think-OFF, item-paired)

```
             pool  n_paired  mean_edi_mt  mean_edi_st  mean_diff_mt_minus_st  boot_ci95_lo  boot_ci95_hi  perm_p_two_sided
         MMLU-dev       237     0.465093     0.370257               0.094836      0.022019      0.168114            0.0123
CommonsenseQA-dev       977     0.842443     0.579191               0.263252      0.226406      0.299659            0.0000
```

## Headline numbers (primary / reference)

- Defense-table ANCHOR ΔAcc (frozen REF): **+65.0 pp** (95% CI [47.5, 80.0]; McNemar p=2.16e-07; rescues=27, harms=1).
- Canonical headline ΔAcc: **+67.5 pp** (95% CI [52.5, 82.5]; McNemar p=1.49e-08).
- System-prompt ΔAcc (frozen): **+25.0 pp** (95% CI [10.0, 40.0]; McNemar p=6.35e-03).
- Best single-layer CAA ΔAcc (frozen): **+10.0 pp** (95% CI [-5.0, 25.0]; McNemar p=3.44e-01).
- CSQA OOD ΔAcc: **+59.7 pp** (95% CI [54.0, 65.3]; McNemar p=2.75e-51).

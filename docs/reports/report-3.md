# Report n°3: the encoder arms on the panel

Pre-registered in `docs/PROTOCOL.md`, "The encoder, and what a verdict is worth (iteration 4)". Development period only; the validation year and the vault stay closed.

## Verdicts

| hypothesis | verdict | rule |
|---|---|---|
| H4 | holds | testable if enc-ols or enc-softmax has a BSS interval above 0; then the one-sided 95 % upper bound and the bootstrap's 95th percentile of mean(ll_ols - ll_softmax) both below 0.005 x climatology-panel's log-loss |
| H5 | does not hold | enc-ols below matrix-select: DM p < 0.05 and the bootstrap interval below 0 |
| H13 | holds | H4 holds and the median wall time per 10^4 decisions of enc-ols over the repeats is at most 0.9 x enc-softmax's, timed on the GPU (the RTX 3060) |

Scored: 1493353 rows over 50626 decision hours (2019-01-01T00 -> 2024-12-30T22), Report n°2's scored rows one for one, paired hour by hour with its series (SHA-256 `691e7ea703fa29f2b857cf353beb5866daf37c3fd8353112bb30173a40aeb843`). Every mean is a mean over hours of the hourly means; intervals are stationary-bootstrap 95 % percentile intervals with Report n°2's seeds.

## Arms

Each encoder arm is the mean of its seeds' probabilities before the floor, then the floor. A row whose window failed takes the arm's climatology (fallbacks).

| arm | log-loss | Brier | BSS vs climatology | ECE | fallbacks |
|---|---|---|---|---|---|
| enc-ols | 1.0659 [1.0623, 1.0696] | 0.6435 [0.6408, 0.6461] | 0.0110 [0.0087, 0.0133] | 0.0064 [0.0044, 0.0130] | 0 |
| enc-softmax | 1.0685 [1.0654, 1.0717] | 0.6455 [0.6433, 0.6478] | 0.0079 [0.0059, 0.0098] | 0.0082 [0.0041, 0.0151] | 0 |
| enc-linear | 1.0728 [1.0699, 1.0756] | 0.6487 [0.6468, 0.6507] | 0.0030 [0.0016, 0.0044] | 0.0113 [0.0036, 0.0203] | 0 |
| climatology-panel (Report n°2) | 1.0756 [1.0726, 1.0786] | 0.6506 [0.6485, 0.6527] | - | 0.0029 [0.0001, 0.0096] | - |
| matrix-select (Report n°2) | 1.0650 [1.0611, 1.0690] | 0.6422 [0.6394, 0.6448] | 0.0130 [0.0108, 0.0156] | 0.0108 [0.0064, 0.0160] | - |

Each seed's log-loss, a description (the model is the ensemble):

| arm | seed 1 | seed 2 | seed 3 | range |
|---|---|---|---|---|
| enc-ols | 1.06746 | 1.06739 | 1.06772 | 0.00033 |
| enc-softmax | 1.07035 | 1.06894 | 1.06932 | 0.00141 |
| enc-linear | 1.07382 | 1.07304 | 1.07364 | 0.00079 |

## Comparisons

Difference = a - b in mean hourly log-loss (negative: a is better).

| comparison | a | b | difference | share of climatology | DM p | one-sided upper 95 % | bootstrap 95 % | bootstrap q95 | identical hours |
|---|---|---|---|---|---|---|---|---|---|
| H4 | enc-ols | enc-softmax | -0.00269 | -0.2498% | 0.0000 | -0.00182 | [-0.00369, -0.00161] | -0.00181 | 0.0% (50626 informative) |
| H5 | enc-ols | matrix-select | +0.00082 | +0.0759% | 0.5198 | +0.00290 | [-0.00147, +0.00313] | +0.00279 | 0.0% (50626 informative) |
| enc-linear vs enc-ols | enc-linear | enc-ols | +0.00697 | +0.6479% | 0.0000 | +0.00839 | [+0.00534, +0.00860] | +0.00832 | 0.0% (50626 informative) |
| enc-linear vs enc-softmax | enc-linear | enc-softmax | +0.00428 | +0.3981% | 0.0000 | +0.00524 | [+0.00326, +0.00535] | +0.00520 | 0.0% (50626 informative) |

## H4

Margin 0.005 x climatology-panel's mean hourly log-loss (1.07556) = 0.005378. Testable: yes (BSS interval above 0: enc-ols True, enc-softmax True). One-sided 95 % upper bound of mean(ols - softmax): -0.001825; bootstrap 95th percentile: -0.001814. Verdict: holds.

Descriptions beside the verdict, not verdicts: the same two bounds and margin on each seed-matched pair (enc-ols seed i against enc-softmax seed i), and on the ensembles in the reversed direction (enc-softmax non-inferior to enc-ols, difference softmax - ols). The verdict above is the registered one, on the ensembles.

| description | a | b | difference | one-sided upper 95 % | bootstrap q95 | both below the margin |
|---|---|---|---|---|---|---|
| seed 1 | enc-ols seed 1 | enc-softmax seed 1 | -0.002892 | -0.001722 | -0.001732 | yes |
| seed 2 | enc-ols seed 2 | enc-softmax seed 2 | -0.001549 | -0.000464 | -0.000545 | yes |
| seed 3 | enc-ols seed 3 | enc-softmax seed 3 | -0.001591 | -0.000682 | -0.000702 | yes |
| reversed | enc-softmax | enc-ols | +0.002687 | +0.003549 | +0.003499 | yes |

## H5

enc-ols against matrix-select (Report n°2's series): does not hold. A limit: the encoder refits yearly and matrix-select monthly: H5 compares two systems, not two attentions at equal cadence.

## H13 and the timing

10 repeats cycling enc-ols, enc-softmax, enc-linear; each: 3 untimed warm-up batches, then 3 timed batches of 4096 rows (12288 rows drawn with the seed 20261030), synchronized before and after each, scaled to 10,000 decisions. Device: NVIDIA GeForce RTX 3060; driver 617.42, CUDA 13.4, torch 2.15.0+cu134; plain fp32 eager, softmax on the math backend, deterministic settings.

| arm | whole pass per 10^4 decisions: min, median, max (s) | attention operator alone, median (s) | memory readout alone, median (s) |
|---|---|---|---|
| enc-ols | 0.3208, 0.3236, 0.3281 | 0.0959 | 0.0162 |
| enc-softmax | 0.7969, 0.8028, 0.8074 | 0.5959 | 0.0173 |
| enc-linear | 0.2207, 0.2261, 0.2282 | 0.0361 | 0.0166 |

Median ratio enc-ols / enc-softmax (the registered rule, a ratio of medians): 0.4031 (at most 0.9). H13: holds. Median of the per-repeat paired ratios, a description: 0.4038. The operator's and the readout's own times are descriptions of the whole pass's split: the operator runs on the GPU on cached Q, K and V, the readout on the CPU in fp64 on cached keys, as the whole pass runs it. The copies of the keys z from the device to the host and of the readout m back belong to the whole pass but to neither of these two timings, nor do the embedding, the layer norms, the MLPs and the head; analytic FLOPs favour OLS by construction when T >> d_head.

## The cost table

Descriptive. GPU columns: NVIDIA GeForce RTX 3060. CPU columns: AMD Ryzen 5 3600 (Linux-6.18.54.1-microsoft-standard-WSL2-x86_64-with-glibc2.43, olsprime_rs 0.1.0). Columns never mix machines.

| arm | parameters | FLOPs per decision | GPU s per 10^4 | GPU peak memory | CPU s per 10^4 | CPU peak memory, whole process | memory's own footprint (analytic) | training GPU-hours |
|---|---|---|---|---|---|---|---|---|
| enc-ols | 21,003 | 2,591,117 | 0.3236 | 574 MiB | - | - | - | 5.419 |
| enc-softmax | 20,995 | 3,535,157 | 0.8028 | 1.01 GiB | - | - | - | 4.349 |
| enc-linear | 20,995 | 2,486,581 | 0.2261 | 596 MiB | - | - | - | 2.922 |
| climatology-panel | 0 | 0 | - | - | 0.0000 | 3.75 GiB | - | - |
| matrix-select | 0 | 43,258 | - | - | 0.0294 | 3.75 GiB | 486 MiB | - |
| matrix-1y | 0 | 41,474 | - | - | 0.0401 | 3.75 GiB | 486 MiB | - |
| softmax-1y | 0 | 17,227,572 | - | - | 3.3016 | 3.75 GiB | 486 MiB | - |
| knn-1y | 0 | 14,599,584 | - | - | 2.0264 | 3.75 GiB | 486 MiB | - |
| matrix-adaptive | 0 | 43,276 | - | - | 0.0289 | 3.75 GiB | 486 MiB | - |
| matrix-stack | 0 | 122,755 | - | - | 0.1062 | 3.75 GiB | 486 MiB | - |
| matrix-cusum | 0 | 43,258 | - | - | 0.0290 | 3.75 GiB | 486 MiB | - |

FLOPs: a multiply-add counts 2. The encoder's formulas are in `report3.encoder_flops` and `attention_flops`, the Matrix arms' in `report3.matrix_flops` (dominant terms, with the mean window sizes of the timed rows). Matrix has no trained parameter: its fit-time choices are selections. Its CPU peak memory is the peak of the whole process, panel and windows included (the panel, the bars, the encoder's windows, torch, the Matrix memory and every arm), where the platform gives it without a subprocess; on Windows it does not. The memory's own footprint is analytic (`report3.matrix_memory_bytes`): 1,591,462 entries x 320 bytes per entry = 486 MiB (entries 243 MiB, two trees of 49,743 nodes 243 MiB). Training GPU-hours sum the fits' wall time; the encoder's peak GPU memory in training: enc-linear 2.00 GiB, enc-ols 1.88 GiB, enc-softmax 2.64 GiB.

## Fits

Fitted on: cuda, in 1 build(s): NVIDIA GeForce RTX 3060, driver 617.42, CUDA 13.4, torch 2.15.0+cu134. The slice losses are the registered early-stopping statistic, not results: each epoch's log-loss on the fit's slice, the rows decided in the 30 days before the fit instant (December rows, the last 30 days of the year before the fit year, at the end of the training years), their labels known 24 h before it; the kept epoch is the first with the lowest.

| arm | seed | year | kept epoch | slice losses | checkpoint SHA-256 |
|---|---|---|---|---|---|
| enc-ols | 1 | 2019 | 5 | 1.0839, 1.0856, 1.0848, 1.0856, 1.0826, 1.0838 | `acc1b3f1edec3006` |
| enc-ols | 1 | 2020 | 5 | 1.0645, 1.0660, 1.0710, 1.0642, 1.0613, 1.0647 | `ed249fe8fcd4048c` |
| enc-ols | 1 | 2021 | 6 | 1.0690, 1.0606, 1.0458, 1.0493, 1.0399, 1.0381 | `d4775c06bef47950` |
| enc-ols | 1 | 2022 | 6 | 1.0977, 1.0973, 1.0842, 1.0851, 1.0884, 1.0816 | `2b2597ce5d79aeb5` |
| enc-ols | 1 | 2023 | 3 | 1.0658, 1.0630, 1.0555, 1.0780, 1.0644, 1.0633 | `455343c781371c5c` |
| enc-ols | 1 | 2024 | 4 | 1.0670, 1.0632, 1.0601, 1.0587, 1.0595, 1.0594 | `7773a596ee4bafed` |
| enc-ols | 2 | 2019 | 4 | 1.0880, 1.0852, 1.0843, 1.0802, 1.0825, 1.0826 | `e3ebb4bf73dc64f6` |
| enc-ols | 2 | 2020 | 6 | 1.0763, 1.0662, 1.0595, 1.0574, 1.0601, 1.0552 | `8a7ab55dd2aecea0` |
| enc-ols | 2 | 2021 | 6 | 1.0706, 1.0697, 1.0682, 1.0409, 1.0376, 1.0362 | `feef4b66f6703080` |
| enc-ols | 2 | 2022 | 4 | 1.1037, 1.0956, 1.0848, 1.0809, 1.0858, 1.0819 | `702834321929299d` |
| enc-ols | 2 | 2023 | 6 | 1.0644, 1.0649, 1.0678, 1.0600, 1.0569, 1.0562 | `a54c61774b8baf9d` |
| enc-ols | 2 | 2024 | 6 | 1.0725, 1.0632, 1.0621, 1.0606, 1.0582, 1.0568 | `2966157ac6591971` |
| enc-ols | 3 | 2019 | 4 | 1.0838, 1.0845, 1.0924, 1.0828, 1.0864, 1.0831 | `026664648709ee8f` |
| enc-ols | 3 | 2020 | 4 | 1.0674, 1.0632, 1.0642, 1.0572, 1.0573, 1.0609 | `f5ac9cd4d0e5be1d` |
| enc-ols | 3 | 2021 | 6 | 1.0659, 1.0566, 1.0467, 1.0436, 1.0376, 1.0369 | `0508f970a9acc1de` |
| enc-ols | 3 | 2022 | 6 | 1.0964, 1.0892, 1.0843, 1.0811, 1.0820, 1.0796 | `601f2d65ee42c97f` |
| enc-ols | 3 | 2023 | 3 | 1.0656, 1.0662, 1.0550, 1.0633, 1.0584, 1.0575 | `4471cbde221019db` |
| enc-ols | 3 | 2024 | 3 | 1.0637, 1.0648, 1.0583, 1.0625, 1.0584, 1.0595 | `d92f22e90c4bb44f` |
| enc-softmax | 1 | 2019 | 6 | 1.0852, 1.0816, 1.0849, 1.0822, 1.0804, 1.0804 | `f9ca35761a2fb864` |
| enc-softmax | 1 | 2020 | 5 | 1.0649, 1.0660, 1.0745, 1.0680, 1.0618, 1.0634 | `bfbe3e0038e27cac` |
| enc-softmax | 1 | 2021 | 6 | 1.0739, 1.0687, 1.0655, 1.0636, 1.0592, 1.0575 | `fadad66dc008e2a7` |
| enc-softmax | 1 | 2022 | 6 | 1.1010, 1.0917, 1.0872, 1.0898, 1.0912, 1.0866 | `af2de5700ceeb916` |
| enc-softmax | 1 | 2023 | 3 | 1.0688, 1.0655, 1.0540, 1.0691, 1.0621, 1.0634 | `f97ace3ed702fa5f` |
| enc-softmax | 1 | 2024 | 6 | 1.0687, 1.0641, 1.0630, 1.0589, 1.0587, 1.0564 | `0b1e6f05db3f2039` |
| enc-softmax | 2 | 2019 | 4 | 1.0884, 1.0839, 1.0826, 1.0814, 1.0828, 1.0821 | `49f4111c26adf257` |
| enc-softmax | 2 | 2020 | 6 | 1.0776, 1.0685, 1.0645, 1.0653, 1.0644, 1.0608 | `5b15f7c995051131` |
| enc-softmax | 2 | 2021 | 6 | 1.0739, 1.0878, 1.0693, 1.0620, 1.0564, 1.0537 | `52d04cb55668a397` |
| enc-softmax | 2 | 2022 | 6 | 1.1021, 1.0903, 1.0858, 1.0850, 1.0865, 1.0806 | `7cd881957b3dc8f7` |
| enc-softmax | 2 | 2023 | 5 | 1.0670, 1.0661, 1.0704, 1.0656, 1.0626, 1.0638 | `cfb2873beeb98045` |
| enc-softmax | 2 | 2024 | 6 | 1.0716, 1.0670, 1.0644, 1.0617, 1.0585, 1.0562 | `d84beb698a742fe0` |
| enc-softmax | 3 | 2019 | 5 | 1.0845, 1.0862, 1.0834, 1.0853, 1.0832, 1.0832 | `0232bad3e461e2bc` |
| enc-softmax | 3 | 2020 | 4 | 1.0708, 1.0639, 1.0617, 1.0577, 1.0595, 1.0592 | `4625f02d99410b4c` |
| enc-softmax | 3 | 2021 | 6 | 1.0710, 1.0698, 1.0667, 1.0599, 1.0515, 1.0500 | `846825fae5ad3323` |
| enc-softmax | 3 | 2022 | 4 | 1.0996, 1.0883, 1.0900, 1.0787, 1.0814, 1.0788 | `d9a636648111871f` |
| enc-softmax | 3 | 2023 | 3 | 1.0695, 1.0715, 1.0597, 1.0707, 1.0619, 1.0621 | `abea0543914de3ed` |
| enc-softmax | 3 | 2024 | 5 | 1.0663, 1.0676, 1.0626, 1.0638, 1.0577, 1.0579 | `f5385fdcc2b03f29` |
| enc-linear | 1 | 2019 | 5 | 1.0850, 1.0815, 1.0848, 1.0822, 1.0805, 1.0805 | `53cba003dd06f126` |
| enc-linear | 1 | 2020 | 5 | 1.0649, 1.0661, 1.0734, 1.0669, 1.0645, 1.0650 | `5d6a77015c864b85` |
| enc-linear | 1 | 2021 | 6 | 1.0737, 1.0695, 1.0700, 1.0684, 1.0680, 1.0679 | `2073d22a127c287a` |
| enc-linear | 1 | 2022 | 6 | 1.1006, 1.0890, 1.0893, 1.0935, 1.0909, 1.0886 | `27c4e551e29cba49` |
| enc-linear | 1 | 2023 | 3 | 1.0691, 1.0691, 1.0593, 1.0692, 1.0642, 1.0633 | `78833a7ff96f3e01` |
| enc-linear | 1 | 2024 | 4 | 1.0697, 1.0669, 1.0630, 1.0610, 1.0631, 1.0622 | `51b024baf1d5eff0` |
| enc-linear | 2 | 2019 | 4 | 1.0885, 1.0840, 1.0828, 1.0817, 1.0832, 1.0825 | `a1bf1df06277825c` |
| enc-linear | 2 | 2020 | 6 | 1.0775, 1.0687, 1.0649, 1.0676, 1.0655, 1.0634 | `910cf478192191ad` |
| enc-linear | 2 | 2021 | 6 | 1.0739, 1.0883, 1.0721, 1.0695, 1.0683, 1.0676 | `cd9bb0102dafd6f1` |
| enc-linear | 2 | 2022 | 3 | 1.1022, 1.0903, 1.0889, 1.0917, 1.0945, 1.0892 | `4c905554240a93f2` |
| enc-linear | 2 | 2023 | 6 | 1.0675, 1.0693, 1.0714, 1.0639, 1.0612, 1.0609 | `24525eb58cc2775a` |
| enc-linear | 2 | 2024 | 6 | 1.0716, 1.0706, 1.0675, 1.0667, 1.0644, 1.0628 | `59b59288ace3e65f` |
| enc-linear | 3 | 2019 | 5 | 1.0847, 1.0856, 1.0834, 1.0852, 1.0831, 1.0832 | `3c4c78677ab7b689` |
| enc-linear | 3 | 2020 | 4 | 1.0712, 1.0649, 1.0638, 1.0624, 1.0636, 1.0640 | `d60e50697d8ff7b0` |
| enc-linear | 3 | 2021 | 6 | 1.0712, 1.0730, 1.0696, 1.0694, 1.0682, 1.0676 | `4f8db8315c54b7a1` |
| enc-linear | 3 | 2022 | 4 | 1.0989, 1.0896, 1.0910, 1.0864, 1.0908, 1.0865 | `dcf16ecd022702f0` |
| enc-linear | 3 | 2023 | 3 | 1.0700, 1.0748, 1.0615, 1.0688, 1.0625, 1.0627 | `dc8e0506828987ce` |
| enc-linear | 3 | 2024 | 5 | 1.0670, 1.0717, 1.0664, 1.0662, 1.0643, 1.0651 | `57e5d37b23987aee` |

## Limits

- H5 compares two systems: the encoder refits yearly and matrix-select monthly.
- The panel H1 of the encoder arms forecasts the sub-panel's cut years with seed 1 before this run: a first look at the encoder's skill there, listed in the change log when it happens.
- Each seed's log-loss is a description; the verdicts read the ensemble.

## Variants

Variants tried so far: 18 before this report, plus the 3 encoder arms: 21. A seed ensemble counts once.

## Provenance

- Git commit: `5f17a76eaab173d89d5eae5ab9384ca39c536321`.
- Code digest (SHA-256 of 84 source and lock files, independent of git): `6b477ae727854102d12bf73255912ed347e2d1b14706a4080ba99d0421d15bad`.
- Data manifest digest: `0ee7ac0f86bd46331ad4f1400e39b8d2df580698f2ebcb849626f8119434ec31`.
- Report n°2's series: `691e7ea703fa29f2b857cf353beb5866daf37c3fd8353112bb30173a40aeb843`; its report2.json: `c9a9eab2ba4e87a24a93cbda69661cb3a0f98efef6310d7bb514310d190dc04f`; panel fingerprint `459798957b8c0d12e82abec46661d9b6d6e96e8961e4799139068d7728167b47`.
- This report's series (`series.npz`), SHA-256: `e8381799c19638ff71bb685876054c476921d705245049be395774d24cae4249`.

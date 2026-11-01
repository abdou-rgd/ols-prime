# Erratum to Report n°3: H13 has no verdict

2026-11-01. [report-3.md](report-3.md) is the generated file of run `runs/2026-10-30-report3/`, commit `5f17a76`. `python -m olsprime.bench.report3 --check` compares it byte for byte with a fresh rendering, so the file stays as the run wrote it. The recorded reading of H13 is the row dated 2026-10-31 in [docs/RESULTS.md](../RESULTS.md): **No verdict: the measurement is invalid.**

## What the generated file prints

Its verdict table:

> | H13 | holds | H4 holds and the median wall time per 10^4 decisions of enc-ols over the repeats is at most 0.9 x enc-softmax's, timed on the GPU (the RTX 3060) |

Its timing section:

> Median ratio enc-ols / enc-softmax (the registered rule, a ratio of medians): 0.4031 (at most 0.9). H13: holds.

`olsprime.bench.report3.verdicts` prints `holds` when H4 holds, the timing is on CUDA and names its GPU, and the ratio of the two medians is at most 0.9 (`Report3Config.h13_ratio`). A `--check` of this run reprints `holds`.

## The record, quoted from `docs/RESULTS.md`

H13, 2026-10-31, result:

> Registered ratio 0.4031 (paired-median ratio 0.4038); bound 0.9. Whole pass, median seconds per 10^4 decisions: `enc-ols` 0.3236, `enc-softmax` 0.8028, `enc-linear` 0.2261. The attention operator alone: 0.0959, 0.5959 and 0.0361 s. Ten repeats, maximum over minimum within 3.4 % for every arm's whole pass, so the gap is not noise. **Invalid as a measurement.** The conditions the protocol names were met, and it names no state of the GPU's memory, but 0.4031 does not measure the arms. The timing stage's own preparation keys the memory over the panel and leaves torch's GPU cache full: 12,926 MiB reserved on a card of 12,288 MiB. Repeated on the same rows, the ratio is 0.3739 as the code stands and 1.0732 with that cache emptied before each protocol call (row "H13's timing, repeated"). Synthetic rows give 1.057 to 1.064 on that same code path.

H13, 2026-10-31, verdict:

> **No verdict: the measurement is invalid.** The rule's arithmetic gives 0.4031 <= 0.9, but the number measures the timing stage's GPU cache, not the arms (row "H13's timing, repeated"). The diagnosis was triggered by the gap with the pre-run synthetic ratio of 1.06 (`docs/ITERATIONS.md`, wave 1), not by the sign of the result. OLS attention is not shown to be cheaper to run than softmax attention

The experiment cell of the row "H13's timing, repeated" (2026-10-31) says: "No score is computed and no verdict reads these repeats." Its result:

> Whole pass, median seconds per 10^4 decisions, `enc-ols` / `enc-softmax` / `enc-linear`: official 0.3236 / 0.8028 / 0.2261, ratio 0.4031; repeat 1, 0.3257 / 0.8711 / 0.2277, ratio 0.3739; repeat 2, 0.1484 / 0.1382 / 0.0945, ratio 1.0732. Operator alone: official 0.0959 / 0.5959 / 0.0361; repeat 1, 0.0998 / 0.5985 / 0.0361; repeat 2, 0.0711 / 0.0605 / 0.0365. Before the first protocol call of repeat 2, torch held 260 MiB allocated and 12,926 MiB reserved, and 370 MiB reserved after emptying. Memory clock 7,301 MHz in 994 of the 1,004 samples at 30 % GPU utilization or more in repeat 2, and 1,237 of 1,262 in repeat 1. The card was full during repeat 1's protocol (nvidia-smi, 11.9 to 12.0 GiB used) and at 1.1 to 3.5 GiB during repeat 2's; the official stage logged neither, so its state is inferred from identical code and data and from repeat 1 reproducing its numbers. In repeat 2 the first 3 to 4 repeats' whole pass is about twice as slow for `enc-ols` and `enc-linear`, while the operator stays flat, and the ratio over repeats 5 to 10 is 1.05. Both repeats ran by day with Windows applications open on the GPU; the official timing ran at night, with no clock log. What emptying the cache changes is measured; how a full cache slows the operators (paging out of an over-committed card is the likely path) is not.

Its verdict:

> Descriptive: the 0.40 is an artefact of the timing stage. One repeat at 1.07 is a description, not a registered measurement

Report n°3's row, on the GPU seconds of the cost table:

> **The GPU wall times of this cost table carry the H13 artefact** (see the rows of H13) and overstate each arm, `enc-softmax` most: read them against the row "H13's timing, repeated". The analytic FLOPs, peak memories and CPU columns are unaffected.

H4 and H5 in the generated file match their rows. H4, verdict:

> **H4 holds as non-inferiority**: `enc-ols` is no worse than `enc-softmax` by more than the margin. The registered rule tests non-inferiority, not superiority; the difference favours `enc-ols`, but that is not what the rule decides

H5, verdict:

> H5 does not hold: no difference was shown, in either direction; the cadences differ

## Where the fix is registered

The fix is registered for iteration 5, before any re-measure. From the change log of [docs/PROTOCOL.md](../PROTOCOL.md), 2026-10-31:

> **For iteration 5, before any re-measure.** The timing stage empties the cache before each protocol call and records the reserved memory, the card's total and the clocks. The bound stays 0.9. The re-measure is a new registered measurement under the amended conditions; its result is recorded whatever it is, with the note that 1.07 had been seen before it ran. A new kernel is a new variant, with its own registration.

From [docs/ITERATIONS.md](../ITERATIONS.md), iteration 4, bolder next step:

> **The timing stage rebuilt.** It empties the cache, and records the reserved memory, the card's total and the clocks beside every ratio. A placebo, a fixed reference kernel timed before and after each protocol call, refuses to write a ratio when the reference drifts: that would have refused 0.4031 at 00:36 UTC, with nobody awake. Then a fused OLS-head kernel as a new registered variant: can OLS attention earn H13 on a kernel worth the name?

## Bolder next step

That rebuilt timing, with the placebo that refuses a drifted ratio, is the measurement that can carry an H13 verdict. The bound stays 0.9. The re-measure is a new registered measurement, recorded whatever it is.

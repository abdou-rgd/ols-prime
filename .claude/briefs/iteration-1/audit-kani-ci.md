# Brief: focused check of the Kani and CI changes (leakage-auditor)

## Goal

Check what changed after re-audit part 3, before the first commit. Is every claim about Kani backed by a log? Can the new CI workflow hide a counterexample?

## What changed

1. `crates/risk/src/proofs.rs`. Five `=> assert!(…)` match arms became blocks, and the `# Cost` section and the `TINY` doc now report the measurements. There is no other change: compare with the pre-change copy at `~/kani-narrow/base/crates/risk/src/proofs.rs` in WSL, using `diff --strip-trailing-cr`.
2. `.github/workflows/kani.yml` is new, and the `kani` job was removed from `ci.yml`.
3. The claims about Kani in:
   - `docs/ITERATIONS.md`: the `crates/risk` and CI rows, the "Kani future-incompat warnings" finding, the Kani lesson and the contracts item under "Carried over";
   - `README.md`: "Getting started";
   - the Verification section of `CLAUDE.md`.
4. `.claude/briefs/iteration-1/kani-narrow.md`, the brief of the narrow-input experiment.

## Evidence (WSL paths)

- `~/kani-runs/latest/summary.tsv`, `~/kani-runs/part2/summary.tsv` and `~/kani-runs/kissat/summary.tsv`, with one log per harness next to each summary.
  - In `latest/`, an `ERROR(exit 1) 0s` line and a `.log` that is a directory mean the second runner claimed that harness. Its result is in `part2/`.
- `~/kani-runs/final/proven.log`: the two proven harnesses re-run on the final code, with the command line of the `proven` job. It may still be running when you start; it began around 02:02.
- `~/kani-narrow/logs/`: the narrow-input experiment and the lot-rounding sweep.
- The scripts that produced them:
  - `run-harnesses.sh`, `run-harnesses-part2.sh` and `run-kissat.sh` in `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\36a2b26e-e1c9-4c1f-b8c1-35bd20889728\scratchpad\kani\`;
  - `~/kani-narrow/run_queue*.sh`.
- `test-workflows.py` in the same scratchpad folder: it parses both workflows and runs the `open` job's script against a fake `cargo`. It passed 7 cases.

## Questions

1. Does every number and verdict in the three docs and in `proofs.rs` match a log? That covers the 2 SUCCESSFUL, the 12 TIMEOUT, the times, the covers, the Kissat result, the 0.1 % circuit claim and the 18× per 4 bits.
2. Is each harness listed in exactly one of `kani.yml`'s two lists, and does `proven` hold exactly the two SUCCESSFUL ones?
3. Can `kani.yml` report success when a harness has a counterexample? Check:
   - the exit-code logic, with 124 as the only code treated as "no verdict";
   - `timeout` and its process group;
   - `bash -e`;
   - the `minutes` validation;
   - the `paths` filters and the concurrency group.
4. Is any claim stronger than the evidence? For example, does anything say or imply that a timed-out harness proves something, or that the bounded domains finish?

## Constraints

Read-only, as always. Do not run Kani: the machine has just finished a 2-hour queue. The re-run in `final/` is the only one in flight.

## Done when

A verdict (pass, or pass with fixes), with each finding located by file and line, and a list of what you checked and found clean.

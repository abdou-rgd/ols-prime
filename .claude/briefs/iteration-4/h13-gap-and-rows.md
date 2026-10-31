# Brief — H13's gap on synthetic data, then the rows of `docs/RESULTS.md`

- **Agent:** `experiment-runner` (sonnet, medium).
- **Written:** 2026-10-31, by the orchestrator.
- **Motto:** we should be more ambitious, in what we test, never in what we claim.
- **Budget:** Abdou's weekly limit is tight. Do these two tasks and stop. Keep tool output short.

Read first: `CLAUDE.md`; `.claude/briefs/iteration-4/resume-after-reset.md`; `notes.md` in that folder (search it, do not read it all); `docs/PROTOCOL.md`, "H4, H5 and H13" and "Size and power of the pass rule".

The scratch folder is `C:\Users\abdou\AppData\Local\Temp\claude\C--Users-abdou-ols-prime\360b1384-377b-4a3a-9ef7-332b5c4fe2de\scratchpad\`. It holds `orchestrator\` (chain and official scripts), `e2e.sh` and `gpu_smoke.sh`. Put your work in a new subfolder, `h13-gap\`.

## Task 1: why did H13's ratio move from 1.06 to 0.40?

**The facts.** Times per 10⁴ decisions, whole pass, seconds:

| | `enc-ols` | `enc-softmax` | `enc-linear` | ratio ols / softmax |
|---|---|---|---|---|
| Wave 1 sizing, synthetic (`docs/ITERATIONS.md`) | 0.142 | 0.134 | 0.093 | 1.06 |
| Report n°3's GPU smokes, synthetic (`notes.md`) | | | | 1.06 to 1.065 |
| Official, real rows, 2026-10-31 00:36 UTC | 0.3236 | 0.8028 | 0.2261 | 0.4031 |

The official operator alone: 0.0959, 0.5959 and 0.0361 s. Ten repeats, min to max within about 1 %, so the gap is not noise. `enc-softmax` got 6 times slower and the two others 2.3 to 2.4 times. H13 holds by its registered rule. The orchestrator will not report 0.40 as a property of OLS attention until the gap has an explanation.

**The question.** Is the gap caused by the rows and the weights (real against synthetic), or by the state of the machine, or by something different between those runs (batch or row count, settings, code)?

**The leading hypothesis, to attack rather than to confirm.** The official timing ran on a quiet GPU, in short bursts between CPU work. An idle RTX 3060 may sit in a lower performance state, with lower memory clocks. `enc-softmax` holds 4,096 × 4 × 64 × 64 fp32 matrices and is memory-bound, so it would pay more than the others. The sizing and the smokes ran right after a fit, with clocks up.

**What to do. Synthetic data only.**
1. Read how the sizing and the smokes timed (`.claude/briefs/iteration-4/encoder-core.md`, `report3.py`, `run_timing`, `SMOKE_*`, `e2e.sh`, `gpu_smoke.sh`). List every difference from the official stage: batch size, rows, repeats, weights, settings, code. A real difference in the conditions may answer the question by itself.
2. Run H13's timing protocol on synthetic rows, from `~/ols-prime-r3` (clean at `5f17a76`, venv `~/ols-r3-gpu`), writing only under your scratch folder, with the official build and settings. Do not change that clone. Point `OLSPRIME_RUNS_DIR` and the data directory at scratch, so that nothing can read the real panel or `runs/`.
3. Log `nvidia-smi --query-gpu=timestamp,pstate,clocks.gr,clocks.mem,utilization.gpu,memory.used` at about 5 Hz during each timing. Run it at least under these two conditions, same code and same rows:
   - **Cold:** the GPU idle for several minutes first, as in the official stage.
   - **Hot:** a separate process keeps the GPU's clocks up during the timing, with a light load that does not compete for it, or the timing right after a training step. Say how you did it.
4. Report the three arms' whole-pass and operator times, the ratio, and the clocks, per condition. If a further single probe would separate the remaining explanations, run it, for example shapes of real size with weights from a fresh seed.

**Constraints.**
- **No new contact.** Never read the real panel, the data store, `runs/2026-10-30-report3/parts` checkpoints or the vault. Never set `OLSPRIME_OPEN_VAULT`. If you think only real rows can settle it, say so and stop: that is the orchestrator's decision.
- Never change `nvidia-smi` settings that need admin or persist (no `-lgc`, `-pl`, `-pm`). Nothing else runs on the GPU meanwhile. The orchestrator keeps the machine quiet until you report.
- Write scripts with a file tool, never through heredocs or a pipe into `python`.
- The chain's shutdown is over, and nothing should shut the machine down. Check `nvidia-smi` first; report anything already using the GPU.

**Done when.** You can state which explanation the measurements support, which they exclude, and what stays unknown. A partial answer stated as partial is a good result; a story the numbers do not force is not.

## Task 2: the rows of `docs/RESULTS.md`

You own these rows. Append them to `docs/RESULTS.md`, uncommitted. Rows are appended, never edited, and follow the format of the existing rows.

1. **Copy** `runs/2026-10-30-report3/report-3.md` to `docs/reports/report-3.md`, unchanged. Report n°2's copy is the model.
2. **Take the four drafts** in `.claude/briefs/iteration-4/results-draft.md` as they stand, except for the power row's TODO: its second run did not produce a byte-identity check. It started on 2026-10-30 at 09:05:23 UTC and stopped at 09:10:17 when its worker pool failed four times at start-up, before any output (`runs/2026-10-30-power-2.log`), and it is not re-run in iteration 4. Say so, and give the first run's facts.
3. **Add these rows,** dated 2026-10-31:
   - **H1 on the encoder arms.** Three verdicts, apart, never folded together: H1 passes; the skill check FAILS on `cheat-stamp` alone (give all three controls); `pos-window` passes (0.6825 [0.6428, 0.7209]). Say the pre-check passed and what it compared, 8 cuts and the 2 purge cuts with their patterns, that exit 3 is the code for "H1 holds and a skill check fails", and that the run was resumed once after its 16-hour timeout with the same folder. Variants: 18, since controls are not variants.
   - **H4**, **H5** and **H13**, one row each, from `runs/2026-10-30-report3/report3.json`. Give the rule's numbers: difference, share of climatology, bounds, margin, testability. H4 is a non-inferiority verdict: do not call it superiority, even though the difference favours `enc-ols`. H5's verdict compares two systems at different refit cadences. H13's row gives the registered ratio and the verdict, and a pointer to Task 1's finding, in one sentence the numbers support. Variants: 21.
   - **Report n°3 (descriptive).** The arms' scores, each seed's log-loss, the cost table. **The training GPU-hours are upper bounds:** the fits ran from 2026-10-30 09:00:54 UTC beside the panel H1 on one GPU. `report-3.md` does not say so, so the row must. Note the first `--check` exit 2 at 00:43:06 UTC was a script error (`--series` and `--sha256` missing) and that the re-run passed at 00:46:52 UTC with 4 files identical, `run.json` excluded. Variants: 21.
   - **H6: not run.** Built, audited once, its fixes unverified, no fit. On the branch `iteration-5-ablations` at `5be70a3`.
4. **Check every number you write** against the run's JSON, with a script, not by eye. Say in your report which numbers you recomputed. If a number disagrees with this brief or a draft, the run's file wins, and you say so.

## Report

To the orchestrator, in English:
- Task 1: the table, the verdict on each explanation, what stays unknown, the commands.
- Task 2: the rows' count, the numbers recomputed, anything that disagreed.
- Everything you did not do, and every process you left running (none, please).
- One bolder next step.

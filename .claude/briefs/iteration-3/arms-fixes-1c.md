# Brief — The six Minor points of audit pass 1c (orchestrator)

- **Who:** the orchestrator, directly: six small, disjoint changes to code the arms engineer wrote, with tests. Then a re-check by the same `leakage-auditor` (pass 1d) before any official run.
- **Why now:** CI cannot run until Abdou re-authenticates gh, so the official runs wait anyway. Fixing every point before they run means the code digest changes once, not after a multi-hour run.

## The points (audit pass 1c, HEAD 91f85d6)

1. **The panel H1 report cannot be told from a smoke run.** `PanelH1Config.registered` (equal to the default config), written to `h1-panel.json` and rendered as the banner Report n°2 already has. Required before the runs.
2. **A cut in a month with no complete row decided by the cut flags the honest system.** The scoped run makes no fit for that month, the original run does. The fit of the cut's month is compared only when a test row of that month is decided by the cut. The two predictions that rest on that fit (σ on the whole test month, selection without purge) take the same condition, so they cannot predict a catch the bench cannot see. Test: the auditor's probe D, every member's bar at a month's first hour dropped, a first-hour cut there.
3. **The CUSUM's s was made from the record's monthly σ, its live values from the fit's σ.** s is now computed as the fit computes its live days: its window, ρ and σ, through `_cusum_readout`. The record no longer carries mid-day readouts. PROTOCOL says so ("with that fit's window, ρ and σ"). Test: s equals the ddof-1 standard deviation of the daily means of standardized residuals read with the fit's own σ.
4. **Report n°2 does not list the days left out at refits, nor the DM kernel.** Each CUSUM fit gets its day left out, or none when that day had no complete 00:00 row; the comparisons table gets the kernel and the lags used, and any fallback note.
5. **A member not yet listed at a cut is left out, unperturbed.** PROTOCOL's panel H1 paragraph names both reasons, and the limit: an arm that read a not-yet-listed member's future bars would not be caught at the cuts before its listing.
6. **Report n°2 does not state its coverage.** `scored` gains the test months' calendar hours and the share scored, with the reason in the report: after an exchange outage the features' 90-bar windows refill, and the labels of the 25 hours before it reach into the gap.

## Constraints

- No change to any registered rule, threshold, grid, seed or period.
- Smoke runs on the real panel exist (the panel H1 to 2019-06; Report n°2 on 30 members to 2020-06). Nothing here is chosen from their numbers; the change log says so.
- Scripts with a file tool, never heredocs. `uv run --no-sync`. No vault read, no vault key.

## Done when

- The full verification passes on Windows, and the WSL2 mirror passes on a clean clone of the new commit.
- `leakage-auditor` re-checks the six points (pass 1d) and keeps its GO.
- Committed on `iteration-3`; `runs.md` names the commit under test once GitHub CI is green.

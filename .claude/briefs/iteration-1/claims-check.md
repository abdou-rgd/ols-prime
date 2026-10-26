# Brief — check the claims the audit left open (iteration 1)

**Role.** `scout`. Read-only: report in your final message and edit no file.

The `leakage-auditor` verified the Binance terms and five arXiv citations, but not the claims below. An earlier scout pass on 2026-10-25 read some of them through a summarizing fetch tool, which can paraphrase numbers. For each claim:
- find the primary source, and give its URL and the date you read it;
- quote at most one short phrase, under 15 words;
- give a verdict: **confirmed**, **wrong** (with the right value and its source), or **unverifiable** (say what you tried and what blocked you).

## 1. Venue fees

Used by `crates/backtest/src/types.rs:90-104`, `docs/PROTOCOL.md:51-57` and `docs/DATA.md:76-85`.
- **Coinbase Advanced, lowest volume tier, for a retail account in France:** maker 0.25 %, taker 0.50 %?
  - The fee page on coinbase.com answered HTTP 403.
  - A Coinbase blog snippet gives 0.25 % / 0.50 % for the EU from 2026-09-16. An older help page gives 0.40 % / 0.60 %.
  - Try Coinbase's help center, the Advanced Trade fee page, the blog post itself and the Advanced Trade API's fee-tier documentation.
  - Say which schedule applies to an EU account on 2026-10-25, and the volume threshold of the lowest tier.
- **Kraken Pro, lowest tier:** maker 0.40 %, taker 0.80 %? Read kraken.com's fee schedule, and say whether an EU schedule differs.
- **Binance spot, VIP 0:** 0.10 % / 0.10 %, and 0.075 % with BNB. Binance is history only for us, so a quick check is enough.

## 2. MiCA authorizations

Used by `docs/DATA.md:87-98` and `CLAUDE.md`.
- ESMA's interim register of crypto-asset service providers: <https://www.esma.europa.eu/sites/default/files/2024-12/CASPS.csv>. Check that each of these rows is authorized and passported to France:

  | Venue | Entity | Authority | Date |
  |---|---|---|---|
  | Coinbase | Coinbase Luxembourg S.A. | CSSF | 2025-06-20 |
  | Kraken | Payward Europe Solutions Limited | Central Bank of Ireland | 2025-06-25 |
  | Bitstamp | Bitstamp Europe S.A. | CSSF | 2025-05-15 |
  | Bitvavo | Bitvavo B.V. | AFM | 2025-06-26 |

- Check that Payward Global Solutions Limited is not passported to France.
- `CLAUDE.md` states that Binance has been closed to French residents since 2026-07-01. Check it against Binance's own announcement, the AMF or the register. Does any Binance entity hold a MiCA authorization covering France?

## 3. The four decision models

Cited in `README.md:16`, `README.md:64` and `docs/PROTOCOL.md:46`:
- Jev (TypeSafe AI): find its primary page;
- Clef (Cloudflare): <https://blog.cloudflare.com/clef-decision-models>;
- Tev1 (Together AI): <https://github.com/togethercomputer/tev1>;
- Microsoft-Decision-1: <https://commandline.microsoft.com/microsoft-decision-1-model-foundry/>.

For each, check:
- the page exists at that URL;
- it dates from autumn 2026 (September to November);
- the model outputs a typed decision, that is a probability or a choice over a fixed set of options;
- for Microsoft-Decision-1 only: it measures how often its decisions flip under small perturbations of the inputs;
- its results are self-reported, with no independent benchmark.

## Constraints

- Primary sources only: the venue's, the regulator's or the vendor's own pages. A news article or an aggregator may point you to one, but it is never the source.
- No sign-in, no account, no form, and nothing downloaded beyond reading the page.
- At most one quote per source, under 15 words.

## Report (short)

- One table: the claim, the verdict, the source URL, the date read, the short quote or number, and a note.
- The corrections we should make, by file and line.
- One bolder next step.

**Done when** every claim has a verdict.

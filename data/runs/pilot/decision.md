# Pilot decision (2026-11-21)

Decided by Salah on 2026-11-21, after the gate report (`report.json`, computed by
`scripts/pilot_report.py` from `calls.jsonl`).

**Gate (PLAN.md 2.7, rule 2): passed.** Every negative and positive was verified
before the freeze (`data/texts/summary.json`). Both central measures were computed
end to end. Per scorer:

| Scorer | Placebo accuracy (40% to 60%) | Invalid (< 2%) | Changed between passes (third pass if > 5%) | USD per call | Grid estimate (24,000 calls) |
|---|---|---|---|---|---|
| haiku-5.5 | 49.2% | 0.67% | 3.7% | 0.000018 | 0.43 |
| sonnet-5.5 | 50.2% | 0.08% | 1.0% | 0.000346 | 8.30 |
| opus-5.5 | 49.5% | 1.50% | 3.3% | 0.000711 | 17.06 |
| gemini-flash | 48.8% | 0.25% | 0.5% | 0.000069 | 1.66 |
| mistral-small | 49.0% | 0.67% | 0.3% | 0.000017 | 0.41 |

**Third pass (PLAN.md 2.5, rule 5): not needed.** No scorer changes its answer on
more than 5% of calls.

**Decision.** The grid starts. Total cap 40 USD for the project (pilot included,
1.39 USD spent), about 35% above the estimate of 27.9 USD for the grid plus the
pilot. The grid reruns the 5 pilot features, so that every grid call comes from one
run.

**Note on the first report.** In the pilot log, the model's reply overwrote the
"text" field of each stored attempt, which holds the index of the text. The raw
log is kept as written. The text index is read back from the call id
(`run.call_fields`), and the code that writes records was fixed before the grid.

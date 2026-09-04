# Multi-source reconciliation agent

**Razorpay AI Buildathon — Track 04, AI Finance Controller**

When a customer pays an online merchant, the money does not go straight to the
merchant's bank. It goes to a payment gateway, which holds it, batches it with
other payments, and pushes a single lump sum to the bank one to three days
later, after deducting its fee and GST on that fee. The merchant is left
holding three records of the same money that look nothing alike: an order
ledger saying ₹10,000 across three orders on Monday, a gateway report showing a
₹200 fee and ₹36 of GST, and a bank statement showing one credit of ₹9,764 on
Wednesday with the narration `NEFT-RAZORPAYSOFT-UTR8842910-SETTLEMENT`. Counts
don't match, amounts don't match, dates don't match, and the linking reference
is buried in junk text — sometimes truncated, sometimes absent. Today a finance
analyst does this by hand in Excel. This project does it automatically, and —
more importantly — tells you exactly what it could not do.

---

## Headline result

Measured on 418 records (193 orders, 190 payments, 35 bank rows) at
`--seed 42`, reproducible with one command.

| Metric | Result |
|---|---|
| **Auto-match rate** | **90.0%** (27 of 30 matchable credits) |
| **Precision** | **100.0%** (27 correct of 27 proposed) |
| **Forced-match errors** | **0** |
| Classification accuracy | 100% (11 of 11 exceptions correctly categorised) |
| Throughput | 9,000-13,000 records/sec (deterministic stages) |
| Cost per run | $0.0017 (4 LLM calls) |

**The number this project is actually about is the third one.** Zero forced
matches means the system never once paired a bank credit with payments it did
not correspond to — including the two credits in the dataset that correspond to
nothing at all, and the two that tie exactly between competing batches.

When a deliberately reckless model is substituted — one that never abstains and
always names every candidate — the verification gate rejects all four of its
proposals, and the numbers move like this:

| | Precision | Forced-match errors |
|---|---|---|
| Without the verification gate | 90.3% | **1** |
| With the verification gate | **100.0%** | **0** |

That single forced match was a proposal against `BNK0016`, a vendor refund that
corresponds to no gateway payment whatsoever. In production that is how a
merchant's books get quietly corrupted.

---

## Quickstart

No API key required. No dependencies beyond the standard library.

```bash
git clone <this repo>
cd recon-agent
python run.py demo          # generate, validate, reconcile, write the report
```

That runs the full deterministic pipeline and writes `runs/<timestamp>/report.md`.
`make demo` does the same thing where `make` is available.

Individual commands:

```bash
python -m recon.cli generate --seed 42 --hard-ratio 0.4 --out data/
python -m recon.cli validate data/
python -m recon.cli run data/ --no-llm
python -m recon.cli sweep data/ --ratios 0.2,0.4,0.6
python -m recon.cli selftest data/     # prove the verification gate works
python run.py test                     # 204 tests, none need an API key
```

To enable the LLM layer, set a key and pass a provider:

```bash
export GROQ_API_KEY=gsk_...            # or ANTHROPIC_API_KEY / GEMINI_API_KEY
python -m recon.cli run data/ --provider groq
```

If the model cannot be reached, the run **exits with an error rather than
silently abstaining** — see "What broke" below for why that matters.

---

## Architecture

**The LLM proposes. Deterministic code disposes.**

```
                  ┌─────────────────────────── deterministic ──────────────────────────┐
  orders.csv  ─┐  │  Stage 1        Stage 2         Stage 3                            │
settlements.csv├─▶│  reference  →   group sum   →   bounded subset sum                 │──┐
   bank.csv   ─┘  │  (regex +       (date window   (meet-in-the-middle,                │  │
                  │   amount check)  + tolerance)   ties → no pick)                    │  │
                  └────────────────────────────────────────────────────────────────────┘  │
                                                                                           ▼
                  ┌──────────────── AI layer ─────────────────┐          ┌─────────────────────────┐
                  │  Stage 4: deterministic top-K retrieval   │          │   residual credits      │
                  │           ↓                               │◀─────────│   (unmatched only)      │
                  │  bounded agentic loop, max 3 iterations   │          └─────────────────────────┘
                  │  actions: propose | widen | more | abstain│
                  └───────────────────┬───────────────────────┘
                                      │  proposal (never a fact)
                                      ▼
                  ┌────────────────────────────────────────────┐
                  │  Stage 5: VERIFICATION GATE                │
                  │  recomputes every amount from source       │
                  │  9 rules · imports nothing from the LLM    │
                  │  accept ──▶ match      reject ──▶ log      │
                  └────────────────────────────────────────────┘
                                      │
                                      ▼
                     Stage 6: classify → report.md + exception list
```

No model output ever becomes a financial fact. The LLM may only *suggest* a
candidate match; independent code then re-verifies that suggestion against the
raw amounts and dates, and rejects it if it does not hold.

---

## Design decisions, and why

**Why LLM-proposes / code-verifies.** LLM arithmetic is unreliable, and a wrong
auto-match silently corrupts a merchant's books — the kind of error nobody
notices for months. The verification gate is cheap, exact and auditable, while
the proposal step is the only part that genuinely needs fuzzy judgement. It also
produces a measurable artifact, the rejection log, that proves the safety layer
does real work. `verify.py` imports nothing from `match_llm.py`, and a test
enforces that: a checker sharing assumptions with the thing it checks is not
checking anything.

**Why bounded subset-sum instead of general.** Subset-sum is NP-complete in
general, but the search here is confined to a single settlement batch of 5–8
payments, capped at 15. Meet-in-the-middle keeps the worst case at ~2^8 rather
than 2^15. At this size a naive powerset would also run instantly — the honest
reason for the bound is a predictable worst case on data whose shape we do not
control, not raw speed.

**Why "do not pick" on ties.** When two settlement batches match a credit
equally well, the system records both candidates and matches neither. Guessing
would raise the match rate and make the software dangerous. Two such credits
appear in the dataset by construction, and both stay unmatched.

**Why synthetic data.** Real settlement data is confidential, and without
ground truth no accuracy claim is checkable. The generator builds the truth
structure in memory first and derives the CSVs from it, so labels are correct by
construction rather than inferred afterwards. `validate.py` then re-derives every
claim in `truth.json` from the CSVs and refuses to pass if any disagree — it runs
11 checks, each verified against a deliberate corruption.

**Why no agent framework.** LangChain, LangGraph and friends would add a
dependency, a layer of indirection, and a set of behaviours I would have to
explain but did not write. The whole system is ~4,000 lines of standard-library
Python (about half of that comments and docstrings). Every decision in it is one I can defend.

**Why a tolerance of 2 paise, even though it costs a correct match.** The gate
rejects a proposal for `BNK0011` whose payment set is *right*, because the credit
carries 5 paise of fee-rounding drift. That is deliberate. A 5-paise gap the
system cannot explain is exactly what an unnoticed short-settlement looks like,
and a gate that waves through unexplained differences is not a gate. It becomes
an exception a human clears in seconds.

---

## What the AI layer actually contributed

Honestly: **on this dataset, +0.0 percentage points.**

The ablation table:

| Configuration | Auto-match rate | Precision | Forced-match errors |
|---|---|---|---|
| Stage 1 only (reference) | 70.0% | 100.0% | 0 |
| Stages 1–2 (+ group sum) | 83.3% | 100.0% | 0 |
| Stages 1–3 (deterministic baseline) | 90.0% | 100.0% | 0 |
| Stages 1–5 (+ LLM, verified) | 90.0% | 100.0% | 0 |

Live run against `openai/gpt-oss-120b` via Groq: 4 calls, 0 failures, 3,795
input tokens, 1,459 output tokens, $0.0017. The model abstained on all four.

That looks like a null result, and the number goes in the report unsoftened.
But look at what the residual actually contained: 2 foreign credits that match
nothing, 2 credits tied exactly between batches, and 1 credit whose correct
answer sits outside tolerance. **Four of the five had no correct answer
available.** The model walked into none of the traps, and its stated reasons
were right:

> `BNK0026` — *"Both settlement STL0017 and STL0028 have identical total net
> amounts matching the credit, making it impossible to uniquely assign the
> credit to a single batch."*

> `BNK0011` — *"Total net of all candidate payments is 3215981 paise, 5 paise
> less than the credit amount 3215986 paise, exceeding the allowed 2-paise
> tolerance."*

A model that had "improved" the match rate here would have done so by guessing.
100% abstention on a residual set that is 80% unanswerable is the correct
behaviour, and it is the behaviour the architecture is built to reward.

---

## Difficulty sweep

A single match rate at one difficulty is easy to over-read, so the same pipeline
is measured at three mixes. `hard_ratio` is the share of settlement batches
carrying a non-clean case.

| hard_ratio | Clean batches | Auto-match rate | Precision | Forced-match errors |
|---|---|---|---|---|
| 0.2 | 25 / 30 | 96.7% | 100.0% | 0 |
| 0.4 | 18 / 30 | 90.0% | 100.0% | 0 |
| 0.6 | 13 / 30 | 86.7% | 100.0% | 0 |

Match rate degrades as the data gets harder, which is expected. Precision and
forced-match errors do not move at all, which is the point.

---

## The exception list is the product

A finance team's real need is *"handle the 90% I don't want to look at, and hand
me a short, correct list of the 10% I do."* Every unresolved record gets a
category and a one-line reason an analyst could act on. All 11 are categorised
correctly against ground truth.

| Record | Category | Reason |
|---|---|---|
| `BNK0004` | `foreign_credit` | Credit of ₹5,931.52 from a non-gateway source; not a settlement. Identify the payer. |
| `BNK0011` | `rounding_drift` | Reference resolves, but amounts differ by 5 paise. Investigate whether the gateway short-settled. |
| `BNK0016` | `foreign_credit` | Credit of ₹3,645.25 from a non-gateway source. |
| `BNK0026` | `ambiguous_amount` | Matches 2 settlement batches equally well (STL0017, STL0028). The system will not guess. |
| `BNK0028` | `ambiguous_amount` | Matches 2 settlement batches equally well. |
| `BNK0022` | `chargeback_reversal` | Debit of ₹5,224.72 reverses an earlier settled payment. |
| `BNK0023`, `BNK0035` | `refund_debit` | Refunds out; match to the refunded order. |
| `ORD00191–193` | `unpaid_order` | Order exists with no payment record. Goods may ship unpaid. |

Adversarial self-test — corrupt a match the pipeline already accepted, and watch
the gate catch it:

```bash
$ python -m recon.cli selftest data/
CAUGHT  amount tampered            → amount_mismatch
CAUGHT  hallucinated payment id    → unknown_payment
CAUGHT  payment claimed twice      → already_claimed
CAUGHT  value date shifted         → date_window
CAUGHT  debit row matched          → not_a_credit
CAUGHT  below confidence threshold → low_confidence
6/6 corruptions caught
```

---

## What broke

Written honestly, because the mistakes were more instructive than the successes.

**1. `weekend_delay` shipped inert, and would have inflated the baseline.**
The T+2/T+3 delay was applied between the order date and `settled_at`, then
`value_date` was set equal to `settled_at`. But the matcher's window is
`[settled_at, settled_at + N]` — the gap that matters is settled → value date,
which was zero for *every* batch. Both `weekend_delay` credits would have
matched trivially and the case would have proven nothing. Caught by reading the
generated data rather than the code. Fixed, plus a validator rule that now fails
if any `weekend_delay` lag is ≤ 2 days.

**2. Stage 1 matched nothing, and the reason was a string comparison.**
The narration regex captured `8973915` while `settlements.csv` stored
`UTR8973915` — the same reference in two formats. Reference matching silently
contributed zero, and stage 2 quietly picked up the slack, so the total looked
plausible. Fixed by comparing digits on both sides. Baseline moved 90.0% → 96.7%
at the time, and stage 1 from 0 to 23 matches.

**3. `ambiguous_amount` could never trigger, and the fix made the number worse.**
The two tied batches were also given clean UTRs, so stage 1 resolved both by
reference before stage 2 ever saw the ambiguity. The case existed in the data and
tested nothing. Removing the reference dropped the headline from 96.7% to 90.0%.
That is the correct direction: the earlier number was inflated by a case that
could not fire.

**4. A failed API call was indistinguishable from a model abstaining.**
The first live run reported `abstention rate: 5/5 (100%)` alongside
`llm_calls: 0`. That reads as a cautious model. It actually meant every call had
thrown an authentication error that was caught, stored in a field nothing
displayed, and reported as an abstention. Abstention rate is a *published
metric* — this was the most dangerous bug in the project, because it produced a
plausible number from a completely broken run. Failures are now counted
separately, printed loudly, and a run where every call fails exits non-zero
rather than writing a report.

**5. The default Groq model had been decommissioned.**
`llama-3.3-70b-versatile` returned `model_not_found`. A retired model id now
reports which models the key can actually use, instead of failing opaquely.

**Dead ends.** The verifier originally trusted a `proposed_sum` field the model
returned; that was removed once it became obvious the gate must recompute
everything from source or it is checking the model against itself. An early plan
to score confidence calibration was dropped — with ~5 proposals per run the
sample is far too small for the curve to mean anything, and publishing it would
have invited a question with no good answer.

---

## Known limitations

- **The dataset is synthetic and self-generated.** Real bank narrations are
  messier than five rotated templates. The match rate would drop on real data;
  the safety properties should not.
- **The LLM layer is unproven on this dataset**, because the residual set is
  almost entirely unanswerable. A dataset with more *hard-but-solvable* cases
  would test it properly — this one mostly tests whether it knows to refuse.
- **Cost figures for `gpt-oss-120b` are estimates.** Token counts are measured;
  the dollar amounts use published rates I could not independently verify. The
  free tier bills nothing.
- **One credit per LLM call, no batching.** Fine at 30 batches, wasteful at
  30,000.
- **The agentic loop is barely exercised** — the model answered on its first
  iteration every time, so `widen_date_window` and `request_more_candidates` are
  tested but not proven useful.
- **Cross-settlement matches are allowed but never occur** in this data, so that
  gate rule is tested only in isolation.

## With another week

1. Build a dataset of hard-but-solvable cases so the LLM layer faces problems
   where guessing correctly is possible — the current residual cannot
   distinguish a good model from a cautious one.
2. Run the same pipeline against 3–4 different models and publish the comparison.
   The gate makes this safe to do: a worse model produces a longer rejection log,
   not a corrupted ledger.
3. Batch LLM calls and measure the cost curve at 10,000+ records.
4. Take one real, anonymised bank statement and measure how far the narration
   regex actually generalises.

---

## Repository layout

```
recon/
  money.py               integer paise, half-up rounding, one display helper
  models.py              frozen dataclasses + the case taxonomy
  generate.py            synthetic data; truth built first, CSVs derived
  validate.py            11 ground-truth consistency checks
  ingest.py              CSV → dataclasses, indexed
  match_deterministic.py stages 1–3, the baseline
  retrieve.py            deterministic top-K candidate selection
  match_llm.py           proposals, caching, audit log, agentic loop
  providers.py           Anthropic / Groq / Gemini adapters
  verify.py              THE GATE — 9 rules, zero LLM awareness
  selftest.py            adversarial corruption of accepted matches
  classify.py            exception categories + analyst-readable reasons
  evaluate.py            scoring against ground truth
  report.py              markdown report writer
  cli.py                 generate | validate | run | sweep | selftest
tests/                   204 tests, none require an API key
data/sample/             a committed dataset you can inspect without running anything
```

Run artifacts land in `runs/<timestamp>/`: `report.md`, `baseline.json`,
`rejections.jsonl`, `llm_calls.jsonl`.

# SPEC.md — Multi-source reconciliation agent

You are building this project with me. Read this entire document before writing any code.

Work **phase by phase**. At the end of each phase, stop, run the stated acceptance check, show me the output, and wait for me to say continue. Do not skip ahead. Do not build Phase 3 while I am reviewing Phase 1.

---

## 1. Why this project exists

This is a submission for the Razorpay AI Buildathon, Track 04 (AI Finance Controller). The brief asks for an agent that closes one finance-ops loop across a 50+ record batch of synthetic data, reporting its match rate and the exceptions it could not resolve. The stated bar is throughput plus measured accuracy plus an honest exception list, with an explicit warning that one cherry-picked match proves nothing.

Read that bar carefully, because it changes engineering priorities:

- **Measured results matter more than features.** A narrow pipeline with an honest metrics table beats a broad one with a demo.
- **The exception list is a deliverable, not a failure.** Unmatched records must be surfaced, categorised and explained.
- **Reproducibility is graded.** A reviewer must be able to clone the repo and reproduce my exact numbers.
- **This will be defended to a technical panel.** Every architectural decision needs a reason I can state out loud.

Optimise for those four things over everything else.

---

## 2. The problem in plain terms

When a customer pays an online merchant, money does not go straight to the merchant's bank. It goes to a payment gateway, which holds it, batches it with other payments, and pushes a single lump sum to the merchant's bank one to three days later, after deducting its own fees and GST on those fees.

This leaves the merchant with three records of the same money that look nothing alike:

| Source | What it says |
|---|---|
| Order ledger | 3 orders, ₹5,000 + ₹3,000 + ₹2,000 = ₹10,000, Monday |
| Gateway settlement report | 3 payments collected, fee ₹200, GST on fee ₹36 |
| Bank statement | **one** credit of ₹9,764 on Wednesday, narration `NEFT-RAZORPAYSOFT-UTR8842910-SETTLEMENT` |

Four things break simultaneously:

1. **Counts don't match.** Many orders collapse into one bank line. There is no row-to-row lookup.
2. **Amounts don't match.** Fees and GST come off mid-flow, so ₹10,000 appears nowhere in the bank statement.
3. **Dates don't match.** Monday's sales land Wednesday, or later across a weekend.
4. **The linking reference is buried in junk text**, sometimes truncated, sometimes absent entirely.

Today a finance analyst does this by hand in Excel. Without it, the merchant cannot answer "did I actually get paid for everything I sold?" — a refund could be double-counted, an order could ship without ever being paid for, or the gateway could short-settle, and nobody would notice.

## 3. What we are building

> Given a merchant's order ledger, gateway settlement report, and bank statement, automatically prove which bank credits correspond to which orders — and clearly flag everything that doesn't add up.

The product framing that matters: a finance team's real need is *"handle the 90% I don't want to look at, and hand me a short, correct list of the 10% I do."* Unmatched records are the product, not the shortfall.

---

## 4. Architecture and the one principle that governs it

**The LLM proposes. Deterministic code disposes.**

No model output ever becomes a financial fact. The LLM may only *suggest* a candidate match; independent, deterministic code then re-verifies that suggestion against the raw amounts and dates, and rejects it if it does not hold. Every rejection is logged with a reason.

This is deliberate and I will be asked to defend it. The reasons:

- LLM arithmetic is unreliable, and a wrong auto-match silently corrupts a merchant's books.
- The verification gate is cheap, exact, and auditable; the proposal step is the only part that needs fuzzy judgement.
- It gives us a measurable artifact — the rejection log — that proves the safety layer does real work.
- The track brief itself argues that verification capacity, not generation speed, is the current bottleneck. The architecture is a direct answer to that claim.

Pipeline:

```
ingest → stage 1 (reference match) → stage 2 (group sum)
       → stage 3 (bounded subset sum) → stage 4 (LLM propose)
       → stage 5 (verification gate) → stage 6 (exception classify)
       → eval → report
```

Stages 1–3 are deterministic and produce the **baseline**. Stages 4–5 are the AI layer. The delta between baseline match rate and final match rate is the headline number of the entire submission, so the baseline must be measured and recorded before the LLM layer is ever called.

---

## 5. Hard constraints

**Language and runtime:** Python 3.11+. Standard library first.

**Allowed dependencies, and nothing else without asking me:**
- `anthropic` (LLM client)
- `pytest` (tests)
- `rich` (CLI output only — optional, drop it if it complicates anything)

**Explicitly forbidden.** Do not add these even if they seem convenient. Each one costs me a question I cannot answer well to a panel:
- No agent frameworks (LangChain, LangGraph, CrewAI, AutoGen)
- No vector database or embeddings
- No web framework, no API server, no frontend, no React, no dashboard
- No pandas (stdlib `csv` is enough at this volume and keeps the data path inspectable)
- No ORM. SQLite via stdlib `sqlite3` only if genuinely needed, and I doubt it is
- No Docker, no CI config, no deployment

**Money handling:**
- All monetary values are **integer paise**. Never float, never Decimal, never rupee strings in storage.
- Convert to rupees only at the display boundary, in one formatting helper.
- Any function that takes an amount takes an `int`. Type-hint it.

**Determinism:**
- Everything seeded. `--seed 42` must reproduce byte-identical data files.
- LLM calls at `temperature=0`.
- LLM responses cached to disk, keyed by a hash of the prompt, so re-runs are free and reproducible.

**Offline mode is mandatory.** `--no-llm` must run the full deterministic pipeline, produce a report, and exit 0 without an API key. A reviewer without credentials must still be able to run the project. This is not optional; treat it as a first-class code path and test it.

---

## 6. Repository layout

```
recon-agent/
├── README.md
├── SPEC.md
├── Makefile
├── pyproject.toml
├── recon/
│   ├── __init__.py
│   ├── money.py           # paise helpers, formatting
│   ├── models.py          # dataclasses: Order, Payment, BankTxn, Match, Exception
│   ├── generate.py        # synthetic data + ground truth
│   ├── validate.py        # ground-truth self-consistency checks
│   ├── ingest.py          # CSV → dataclasses, normalisation
│   ├── match_deterministic.py
│   ├── match_llm.py
│   ├── verify.py          # the gate
│   ├── classify.py        # exception categorisation
│   ├── evaluate.py        # scoring against truth
│   ├── report.py          # markdown report writer
│   └── cli.py
├── tests/
├── data/                  # generated, gitignored except a committed sample
└── runs/                  # per-run artifacts, gitignored
```

---

## 7. Data model

### `orders.csv`
| Column | Type | Notes |
|---|---|---|
| `order_id` | str | `ORD00001` |
| `customer_name` | str | |
| `invoice_paise` | int | |
| `created_at` | ISO date | |
| `status` | str | `paid` \| `unpaid` \| `refunded` |

### `settlements.csv` (the gateway report)
| Column | Type | Notes |
|---|---|---|
| `payment_id` | str | `PAY00001` |
| `order_id` | str | FK to orders |
| `gross_paise` | int | equals `invoice_paise` |
| `fee_paise` | int | 2% of gross, rounded half-up |
| `gst_paise` | int | 18% of fee, rounded half-up |
| `net_paise` | int | `gross - fee - gst` |
| `settlement_id` | str | `STL0001`, groups payments into a batch |
| `settled_at` | ISO date | |
| `utr` | str | may be empty for some cases |

### `bank.csv`
| Column | Type | Notes |
|---|---|---|
| `txn_id` | str | `BNK0001` |
| `value_date` | ISO date | |
| `narration` | str | messy free text |
| `credit_paise` | int | 0 if debit row |
| `debit_paise` | int | 0 if credit row |
| `balance_paise` | int | running, must be internally consistent |

### `truth.json`
```json
{
  "seed": 42,
  "hard_ratio": 0.4,
  "generated_at": "2026-09-01T18:00:00",
  "matches": [
    {
      "bank_txn_id": "BNK0007",
      "payment_ids": ["PAY00031", "PAY00032"],
      "case": "clean_batch"
    }
  ],
  "unmatchable_bank": [
    {"bank_txn_id": "BNK0019", "case": "foreign_credit"}
  ],
  "unpaid_orders": [
    {"order_id": "ORD00088", "case": "unpaid_order"}
  ]
}
```

---

## 8. Phase 1 — Generator and validator

**This is the most important phase. Everything downstream is meaningless without correct ground truth.**

### Critical design rule

**Generate the truth structure first, in memory, then derive the CSVs from it.** Decide the settlement batches and their case labels, then emit rows. Do not generate CSVs and try to label them afterwards — the labels will be subtly wrong and I will lose Phase 3 debugging the scorer instead of the matcher.

### Volumes (at defaults)
- ~150 orders
- ~150 payments
- ~20 settlement batches
- ~25 bank rows

Comfortably clears the 50+ record bar without slowing the eval loop.

### Case mix

Target roughly 60% clean. If everything is hard the match rate is meaningless; if everything is easy the project proves nothing. Every generated batch carries a `case` label, and these labels become the exception taxonomy later.

| Case label | Behaviour | Count at defaults |
|---|---|---|
| `clean_batch` | 8–12 payments, one credit, clean UTR present in narration | ~60% of batches |
| `weekend_delay` | Settled T+3 instead of T+2, falls outside a naive 2-day window | 2 batches |
| `mangled_utr` | Reference truncated to first 8 chars, or stray digits spliced in | 2 batches |
| `missing_utr` | No reference at all; must match on amount + date window alone | 2 batches |
| `ambiguous_amount` | Two batches with identical net, same value date — a genuine tie | 1 pair |
| `partial_batch` | One payment withheld under review; credit = group total **minus** that payment. This is what forces real subset-sum. | 2 batches |
| `rounding_drift` | 1–2 paise off from fee rounding | 2 batches |
| `refund_debit` | Refund arrives as a separate debit 2 days later | 2 rows |
| `chargeback_reversal` | Debit reversing an earlier credit | 1 row |
| `foreign_credit` | Non-gateway credit (a vendor refund). **Matches nothing.** | 2 rows |
| `unpaid_order` | Order exists, no payment ever recorded. Must surface as exception. | 3 orders |

`foreign_credit` and `unpaid_order` are the two most valuable cases in the dataset. They are the only way to prove the system says "I don't know" instead of forcing a match. Forced matches are what makes reconciliation software dangerous in production. Do not omit them, do not let any matcher stage quietly pair them.

### Narration templates

Rotate these so the regex has real work:

```
NEFT-RAZORPAYSOFT-UTR{utr}-SETTLEMENT
IMPS/{utr}/RAZORPAY/COLLECTION
RTGS RZPYSOFT {utr} NET STLMT
UPI-RAZORPAY-{utr}
{utr} RAZORPAYSOFTWA
```

For `foreign_credit`, use something structurally different that must not regex-match: `NEFT-ACMESUPPLIES-INV4471`.

### CLI
```
python -m recon.cli generate --seed 42 --hard-ratio 0.4 --out data/
python -m recon.cli validate data/
```

`--hard-ratio` scales the proportion of non-clean cases. Make it real — I want to report match rate at three difficulty mixes in the final table, which is far more convincing than a single number.

### `validate.py` must assert
- Every `clean_batch` credit equals the exact sum of its payments' `net_paise`
- Every `partial_batch` credit equals the group sum minus exactly the withheld payment
- For every payment: `gross - fee - gst == net`
- Every `payment_id` and `order_id` referenced in `truth.json` exists in the CSVs
- No `payment_id` appears in more than one truth match
- Bank balances are cumulative and arithmetically consistent down the statement
- No `foreign_credit` narration matches the reference-extraction regex
- Counts of each `case` match what was requested

Exit non-zero with a clear message on any failure.

### Phase 1 acceptance
```
python -m recon.cli generate --seed 42 --out data/ && python -m recon.cli validate data/
```

Four files produced, validator passes clean, and re-running with the same seed produces byte-identical files. Show me a `diff` proving the second run is identical.

---

## 9. Phase 2 — Deterministic matcher and baseline

Do **not** touch the LLM in this phase.

### Stage 1 — reference match
Extract candidate references from narration by regex. Match against payment `utr` / `settlement_id`. On hit, verify the amount before accepting (a reference match with a wrong amount is a data problem, not a match — log it as an exception).

### Stage 2 — group sum
Group unclaimed payments by `settlement_id`. Compare each group's `sum(net_paise)` against unmatched bank credits within a configurable date window (default: value_date within `[settled_at, settled_at + 3 days]`) and an amount tolerance (default: 2 paise, to absorb rounding drift).

If two groups tie for one credit, **do not pick one**. Mark it ambiguous and leave it for later stages. Guessing here is exactly the failure mode we are trying to avoid.

### Stage 3 — bounded subset sum
For credits still unmatched, take the best candidate settlement group and search subsets of it that sum to the credit amount within tolerance. This is what resolves `partial_batch`.

Bound it hard:
- Only search within a single settlement group (typically 8–12 payments)
- Cap group size at 15; beyond that, skip to the LLM stage
- Use a meet-in-the-middle or DP approach, not naive `itertools` over the powerset
- If more than one subset sums correctly, mark ambiguous — do not pick

### Eval harness

Define these precisely and implement them exactly:

- A **match** is a pair `(bank_txn_id, frozenset(payment_ids))`.
- **True positive:** a proposed match whose payment set is *exactly equal* to the truth set for that bank transaction.
- **Partial match:** payment sets overlap but are not equal. Counts as a false positive for strict precision, **but is reported separately** — it is diagnostically useful and hiding it would be dishonest.
- **False positive:** any proposed match that is not a true positive, including any match proposed for a `foreign_credit` row.
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (total truth matches)
- **Auto-match rate** = matched bank transactions / matchable bank transactions
- **Forced-match errors** = matches proposed against `unmatchable_bank` entries. Report this as its own line. It should be zero, and if it isn't that is the most important number on the page.

### Phase 2 acceptance
```
python -m recon.cli run data/ --no-llm
```

Prints a metrics table with all of the above. Record these numbers — they are the **baseline** and every later comparison is against them. Do not let the baseline numbers get overwritten or lost when the LLM layer lands; persist them to `runs/<timestamp>/baseline.json`.

---

## 10. Phase 3 — LLM proposal layer

### Retrieval is deterministic
For each residual unmatched credit, deterministically retrieve the top-K candidate payments (K default 20) by date window and amount plausibility. **The LLM never sees the full dataset** — this bounds cost, latency and hallucination surface.

### The call
- Model: `claude-sonnet-4-6`
- `temperature=0`
- Structured JSON response. Prompt the model to return only JSON, no prose, no markdown fences. Strip fences defensively anyway before parsing.
- One credit per call. No batching in the first version.

Response schema:
```json
{
  "proposed_payment_ids": ["PAY00031", "PAY00032"],
  "confidence": 0.87,
  "reasoning": "Sum of net amounts equals credit within 1 paise; both settled T+2 before value date; narration fragment UTR8842 matches settlement STL0007.",
  "abstain": false
}
```

**The model must be able to abstain.** Make `abstain: true` a first-class, explicitly encouraged option in the prompt, for when no candidate set is defensible. A model that always answers is worse than useless here. Track the abstention rate as a reported metric.

### Caching and audit
- Cache every response to `runs/cache/<sha256-of-prompt>.json`
- Append every call to `runs/<timestamp>/llm_calls.jsonl` with: prompt hash, credit id, candidates offered, raw response, parsed result, latency, input/output tokens, estimated cost

That JSONL is the audit trail. The track brief cares about audit trails; this is ours, and it costs almost nothing to produce.

---

## 11. Phase 4 — Verification gate

**This is the most important module in the repository.** Write it as pure functions with no LLM awareness whatsoever — it must be independently testable and must not import `match_llm`.

Every LLM proposal is re-checked from raw source data:

1. Do the proposed payments' `net_paise` sum to the credit amount within tolerance? (recompute from source, never trust the model's arithmetic)
2. Is every proposed `payment_id` real and currently unclaimed?
3. Is each payment's `settled_at` inside the allowed window relative to the credit's `value_date`?
4. Are all proposed payments from the same `settlement_id`? If not, require a higher confidence threshold and flag it.
5. Is `confidence` above the acceptance threshold (default 0.7)?
6. Does accepting this create a conflict with any already-accepted match?

Any failure → **reject**, push the credit to exceptions, and append to `runs/<timestamp>/rejections.jsonl` with the specific rule that failed and the numbers involved.

The rejection log is the single best artifact in the submission. It lets me say, on camera: *"the model proposed 27 matches, my verifier rejected 5, here are all five and exactly why."* Make sure it is human-readable and includes the actual amounts, not just a rule name.

---

## 12. Phase 5 — Exception classification and report

Everything unmatched gets classified into a named category with a plain-English, one-line reason a finance analyst could act on. Use the LLM for categorisation only — never to invent a match — and always fall back to a deterministic `unclassified` bucket if the call fails or `--no-llm` is set.

Categories should mirror the generator's case labels plus an `unknown` bucket, so I can report classification accuracy against truth as a bonus metric.

### `report.py` emits `runs/<timestamp>/report.md`

Required sections:

1. **Run header** — seed, hard-ratio, record counts, wall-clock time, git commit hash
2. **Headline metrics table** — auto-match rate, precision, recall, forced-match errors
3. **Ablation table** — the number that matters most:

| Configuration | Auto-match rate | Precision | Forced-match errors |
|---|---|---|---|
| Stage 1 only (reference) | | | |
| Stages 1–2 (+ group sum) | | | |
| Stages 1–3 (deterministic baseline) | | | |
| Stages 1–5 (+ LLM, verified) | | | |

4. **Difficulty sweep** — the same headline metrics at `--hard-ratio` 0.2 / 0.4 / 0.6
5. **Verification gate summary** — proposals made, accepted, rejected, rejections broken out by rule
6. **Full exception list** — every unresolved record, its category, and its reason. Complete, not truncated.
7. **Throughput and cost** — records/second, LLM calls per 100 records, estimated cost per 1,000 records

Section 6 must never be truncated for tidiness. The honest exception list is the deliverable.

---

## 13. CLI surface

```
python -m recon.cli generate --seed 42 --hard-ratio 0.4 --out data/
python -m recon.cli validate data/
python -m recon.cli run data/ [--no-llm] [--confidence-threshold 0.7] [--out runs/]
python -m recon.cli sweep data/ --ratios 0.2,0.4,0.6
```

Makefile targets: `make demo` (generate + validate + run, no LLM, end to end from a clean clone), `make demo-llm`, `make test`.

`make demo` working from a fresh `git clone` with no API key is a hard requirement. A reviewer who hits an error there stops reviewing.

---

## 14. Testing

`pytest`, and keep it focused. Priority order:

1. **Verification gate** — highest value. Test each rejection rule in isolation, including a proposal whose amounts don't sum, one referencing a claimed payment, one outside the date window, one below the confidence threshold.
2. **Money helpers** — rounding half-up, paise/rupee conversion round-trips.
3. **Subset-sum** — known inputs, known outputs, including the no-solution and multiple-solution cases.
4. **Generator determinism** — same seed, identical output.
5. **Validator** — deliberately corrupt a truth file and assert it is caught.
6. **End-to-end `--no-llm`** — runs clean and produces a report.

Mock the LLM in all tests. No test may require an API key.

---

## 15. README requirements

Write this last, but write it properly — it is a graded deliverable.

- One-paragraph problem statement in plain language (the "three records of the same money" framing)
- Architecture diagram (ASCII or Mermaid) showing the stage pipeline and the propose/verify split
- **The headline result in the first screenful.** Baseline vs final match rate, precision, forced-match errors.
- Quickstart: exact commands, working from a clean clone with no API key
- Design decisions with reasons: why LLM-proposes/code-verifies, why bounded subset-sum instead of general, why synthetic data, why no agent framework
- **A "what broke" section**, written honestly — the application form asks about this. Real bugs, real dead ends, what I changed. Do not sand this down.
- Known limitations and what I would do with another week

---

## 16. Build order and how to work with me

| Phase | Content | Stop and show me |
|---|---|---|
| 1 | Generator, truth, validator | Determinism diff + validator passing |
| 2 | Ingest, stages 1–3, eval harness | The baseline metrics table |
| 3 | LLM proposer, caching, call log | 5 sample proposals with reasoning |
| 4 | Verification gate + its tests | The rejection log with real rejections in it |
| 5 | Classification, report, sweep | The full `report.md` |
| 6 | README, tests, cleanup | `make demo` from a clean clone |

Rules for how you work:

- **Stop at every phase boundary.** Show output, wait for me.
- **Never invent scope.** If something seems missing from this spec, ask before building it.
- **Never let a match through unverified**, even temporarily as a shortcut. That path becomes permanent.
- **Never delete or silently soften a failing metric.** If the number is bad, the number is bad and it goes in the report.
- If you disagree with a decision in this spec, say so and explain why before implementing. I have to defend all of this to a panel, so I would rather argue now than discover the weakness on camera.

Start with Phase 1.

---

## Appendix — Deviations from this spec as built

This spec was written before implementation. Eight internal contradictions were found during planning and resolved with the author before any code was written; the reasoning for each is recorded in `SPEC_AMENDMENTS.md` (not committed). Summary of what the built system does differently:

| # | Spec says | Built as | Why |
|---|---|---|---|
| 1 | ~20 batches, "roughly 60% clean" | **30 batches**, 18 clean | 20 batches minus the 12 named non-clean cases leaves 8 clean = 40%, contradicting the 60% target |
| 2 | ~150 payments, 8–12 per batch | **~190 payments**, 5–8 per batch | 20 × 8–12 = 160–240, which cannot equal ~150 |
| 3 | `rounding_drift` is 1–2 paise off | Drift lives on the **bank credit**, not the payment row | Otherwise it breaks the validator's own `gross - fee - gst == net` rule |
| 4 | Drift of 1–2 paise | Drift of **1–5 paise** | 1–2 paise is fully absorbed by the 2-paise stage-2 tolerance, so the case would exercise nothing |
| 5 | `partial_batch` withholds one payment | Withheld payment's net must be **unique within its group** | Otherwise two subsets tie, the matcher correctly reports ambiguity, and the ground truth becomes unmatchable |
| 6 | Auto-match rate = matched / matchable | Numerator counts only **correct** matches | As written, a matcher that guesses on every row scores 100% |
| 7 | `truth.json` has matches / unmatchable / unpaid | Adds **`debit_rows`** + order-side metrics | Unpaid orders and debit rows could never appear as credit matches, so 6 records affected no metric |
| 8 | Forced-match errors reported separately | Reported, with the **overlap with FP stated** | They are a strict subset of false positives, not a separate population |
| — | Model `claude-sonnet-4-6` | `claude-sonnet-5` default; **Groq/Gemini adapters added** | No Anthropic key was available. Leaving the LLM ablation row unmeasured was the worse outcome |
| — | `make demo` | Makefile delegates to `run.py` | `make` is not present on all machines, notably Windows |

Two additions beyond the spec, agreed before implementation:

- **`recon/selftest.py`** — deliberately corrupts matches the pipeline already accepted and proves the verification gate rejects each one. Run with `python -m recon.cli selftest data/`.
- **Cash-position reconciliation** in report section 7, addressing the second half of the track's "run the books and the cash position".
- **A bounded agentic loop** in the LLM layer (max 3 iterations; the model selects `propose_match` / `widen_date_window` / `request_more_candidates` / `abstain`), so that "agent" is defensible without giving up determinism. Every action still passes through the same verification gate.

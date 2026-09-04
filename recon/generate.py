"""Synthetic data generator.

Design rule that governs this whole module: **build the truth structure first,
in memory, then derive the CSVs from it.** Deciding batches and case labels up
front and emitting rows afterwards means the labels are correct by
construction. Generating CSVs and labelling them after the fact produces
subtly wrong ground truth, which silently invalidates every downstream metric.

Determinism: a single seeded ``random.Random`` instance drives every choice.
Same seed in, byte-identical files out.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

from .models import Case
from .money import net_for

# --- Tunables -------------------------------------------------------------
# Volumes resolved during planning: 30 batches gives 18 clean (60%) alongside
# the 12 non-clean batches the case mix requires. The SPEC's original 20
# batches could only ever reach 40% clean, contradicting its own target.
DEFAULT_BATCHES = 30
PAYMENTS_PER_BATCH = (5, 8)
UNPAID_ORDERS = 3
START_DATE = date(2026, 1, 5)  # a Monday

#: Non-clean batch cases and how many batches each consumes at hard_ratio=0.4.
HARD_BATCH_MIX: dict[Case, int] = {
    Case.WEEKEND_DELAY: 2,
    Case.MANGLED_UTR: 2,
    Case.MISSING_UTR: 2,
    Case.AMBIGUOUS_AMOUNT: 2,  # "1 pair" == 2 batches sharing a net + date
    Case.PARTIAL_BATCH: 2,
    Case.ROUNDING_DRIFT: 2,
}

NARRATION_TEMPLATES = [
    # References already carry a "UTR" prefix, so the template must not add a
    # second one (it produced "UTRUTR8842910" before this was noticed).
    "NEFT-RAZORPAYSOFT-{utr}-SETTLEMENT",
    "IMPS/{utr}/RAZORPAY/COLLECTION",
    "RTGS RZPYSOFT {utr} NET STLMT",
    "UPI-RAZORPAY-{utr}",
    "{utr} RAZORPAYSOFTWA",
]
# Structurally different so the reference regex cannot match it.
FOREIGN_NARRATIONS = [
    "NEFT-ACMESUPPLIES-INV4471",
    "NEFT-VENDORREFUND-BILL2290",
]
NO_REF_NARRATION = "NEFT-RAZORPAYSOFT-SETTLEMENT-BULK"

FIRST_NAMES = ["Aarav", "Diya", "Rohan", "Ananya", "Vihaan", "Ishita", "Kabir",
               "Meera", "Arjun", "Saanvi", "Aditya", "Riya", "Karan", "Nisha",
               "Vikram", "Pooja", "Rahul", "Sneha", "Manav", "Tara"]
LAST_NAMES = ["Sharma", "Iyer", "Patel", "Reddy", "Nair", "Gupta", "Bose",
              "Menon", "Joshi", "Rao", "Khanna", "Desai", "Bhat", "Chopra"]


@dataclass
class PlannedPayment:
    payment_id: str
    order_id: str
    gross_paise: int
    fee_paise: int
    gst_paise: int
    net_paise: int
    customer_name: str
    order_date: date


@dataclass
class PlannedBatch:
    """A settlement batch plus everything needed to emit its bank credit."""

    settlement_id: str
    case: Case
    payments: list[PlannedPayment]
    settled_at: date
    value_date: date
    utr: str
    narration_utr: str  # what actually appears in narration (may be mangled/"")
    template_idx: int
    withheld_payment_id: str | None = None
    drift_paise: int = 0

    def claimed_payments(self) -> list[PlannedPayment]:
        """Payments the credit actually covers (partial_batch withholds one)."""
        if self.withheld_payment_id is None:
            return self.payments
        return [p for p in self.payments if p.payment_id != self.withheld_payment_id]

    def credit_paise(self) -> int:
        return sum(p.net_paise for p in self.claimed_payments()) + self.drift_paise


@dataclass
class Plan:
    """The complete in-memory truth structure. CSVs are derived from this."""

    seed: int
    hard_ratio: float
    batches: list[PlannedBatch] = field(default_factory=list)
    unpaid_orders: list[tuple[str, str, int, date]] = field(default_factory=list)
    foreign_credits: list[tuple[date, str, int]] = field(default_factory=list)
    refund_debits: list[tuple[date, int, str]] = field(default_factory=list)
    chargebacks: list[tuple[date, int, str]] = field(default_factory=list)


def _business_day_offset(start: date, days: int) -> date:
    """Advance ``days`` calendar days, then skip forward off a weekend."""
    d = start + timedelta(days=days)
    while d.weekday() >= 5:  # Sat/Sun -> next Monday
        d += timedelta(days=1)
    return d


def _make_utr(rng: random.Random) -> str:
    return "UTR%d" % rng.randint(1000000, 9999999)


def _mangle(utr: str, rng: random.Random) -> str:
    """Truncate to 8 chars, or splice a stray digit in."""
    if rng.random() < 0.5:
        return utr[:8]
    pos = rng.randint(4, len(utr) - 1)
    return utr[:pos] + str(rng.randint(0, 9)) + utr[pos:]


def _resolve_case_mix(hard_ratio: float, n_batches: int) -> list[Case]:
    """Scale the hard-case mix by ``hard_ratio`` and pad with clean batches.

    ``hard_ratio`` is the target share of non-clean batches. At the 0.4
    default this reproduces the SPEC's stated counts exactly; other ratios
    scale each hard case proportionally so the difficulty sweep is real.
    """
    baseline_hard = sum(HARD_BATCH_MIX.values())  # 12 at defaults
    scale = (hard_ratio * n_batches) / baseline_hard if baseline_hard else 0.0
    cases: list[Case] = []
    for case, count in HARD_BATCH_MIX.items():
        n = round(count * scale)
        # ambiguous_amount is a *pair*; a single one cannot create a tie.
        if case is Case.AMBIGUOUS_AMOUNT:
            n = (n // 2) * 2
        cases.extend([case] * min(n, n_batches))
    cases = cases[:n_batches]
    cases.extend([Case.CLEAN_BATCH] * (n_batches - len(cases)))
    return cases


def build_plan(seed: int, hard_ratio: float,
               n_batches: int = DEFAULT_BATCHES) -> Plan:
    """Build the complete truth structure in memory. No CSV emission here."""
    rng = random.Random(seed)
    plan = Plan(seed=seed, hard_ratio=hard_ratio)

    cases = _resolve_case_mix(hard_ratio, n_batches)
    # Deterministic ordering: sort by label, then one seeded shuffle.
    cases.sort(key=lambda c: c.value)
    rng.shuffle(cases)

    order_n = 0
    payment_n = 0
    used_utrs: set[str] = set()
    ambiguous_pending: PlannedBatch | None = None

    for i, case in enumerate(cases, start=1):
        settlement_id = "STL%04d" % i
        order_day = _business_day_offset(START_DATE, rng.randint(0, 45))

        n_payments = rng.randint(*PAYMENTS_PER_BATCH)
        payments: list[PlannedPayment] = []
        for _ in range(n_payments):
            order_n += 1
            payment_n += 1
            gross = rng.randint(50_000, 900_000)  # Rs 500 - Rs 9,000
            fee, gst, net = net_for(gross)
            payments.append(PlannedPayment(
                payment_id="PAY%05d" % payment_n,
                order_id="ORD%05d" % order_n,
                gross_paise=gross, fee_paise=fee, gst_paise=gst, net_paise=net,
                customer_name=rng.choice(FIRST_NAMES) + " " + rng.choice(LAST_NAMES),
                order_date=order_day,
            ))

        # The gateway settles an order T+1 after it is placed.
        settled_at = _business_day_offset(order_day, 1)

        # Money then lands in the bank T+2 after settlement -- or T+3 for
        # weekend_delay, which pushes value_date outside a naive 2-day window.
        # This settled_at -> value_date gap is the one stage 2 actually tests
        # (its window is [settled_at, settled_at + 3 days]), so the delay must
        # live here and not upstream of settled_at.
        bank_lag = 3 if case is Case.WEEKEND_DELAY else 2
        value_date = settled_at + timedelta(days=bank_lag)

        utr = _make_utr(rng)
        while utr in used_utrs:
            utr = _make_utr(rng)
        used_utrs.add(utr)

        narration_utr = utr
        if case is Case.MANGLED_UTR:
            narration_utr = _mangle(utr, rng)
        elif case is Case.MISSING_UTR:
            narration_utr = ""

        batch = PlannedBatch(
            settlement_id=settlement_id, case=case, payments=payments,
            settled_at=settled_at, value_date=value_date, utr=utr,
            narration_utr=narration_utr,
            template_idx=rng.randrange(len(NARRATION_TEMPLATES)),
        )

        if case is Case.PARTIAL_BATCH:
            # Withhold one payment. Its net MUST be unique within the group,
            # else two different subsets sum to the credit, the matcher
            # correctly reports ambiguity, and our own ground truth becomes
            # unmatchable. Enforced here, asserted in validate.py.
            nets = [p.net_paise for p in payments]
            unique = [p for p in payments if nets.count(p.net_paise) == 1]
            if not unique:  # pragma: no cover - vanishingly rare
                payments[0] = _rebuild_with_gross(payments[0],
                                                  payments[0].gross_paise + 7919)
                unique = [payments[0]]
            batch.withheld_payment_id = rng.choice(unique).payment_id

        if case is Case.ROUNDING_DRIFT:
            # Drift lives on the bank credit, NOT inside the payment rows --
            # the per-payment gross-fee-gst==net identity must stay exact.
            # 1-5 paise, so it often exceeds the 2-paise stage-2 tolerance and
            # is therefore visible in a zero-tolerance ablation.
            batch.drift_paise = rng.choice([-5, -3, -1, 1, 3, 5])

        if case is Case.AMBIGUOUS_AMOUNT:
            if ambiguous_pending is None:
                ambiguous_pending = batch
            else:
                # Force a genuine tie: identical net total and value date.
                _equalise_nets(ambiguous_pending, batch)
                batch.value_date = ambiguous_pending.value_date
                batch.settled_at = ambiguous_pending.settled_at
                ambiguous_pending = None

        plan.batches.append(batch)

    # Order-side exceptions: orders that exist with no payment ever recorded.
    for _ in range(UNPAID_ORDERS):
        order_n += 1
        plan.unpaid_orders.append((
            "ORD%05d" % order_n,
            rng.choice(FIRST_NAMES) + " " + rng.choice(LAST_NAMES),
            rng.randint(50_000, 900_000),
            _business_day_offset(START_DATE, rng.randint(0, 45)),
        ))

    # Bank rows that match nothing. These prove the system can say "I don't
    # know" instead of forcing a match -- the most valuable cases generated.
    for narration in FOREIGN_NARRATIONS:
        plan.foreign_credits.append((
            _business_day_offset(START_DATE, rng.randint(5, 50)),
            narration, rng.randint(100_000, 800_000),
        ))
    for _ in range(2):
        src = rng.choice(plan.batches)
        victim = rng.choice(src.payments)
        plan.refund_debits.append((
            _business_day_offset(src.value_date, 2), victim.net_paise,
            "NEFT-RAZORPAY-REFUND-" + victim.order_id,
        ))
    src = rng.choice(plan.batches)
    victim = rng.choice(src.payments)
    plan.chargebacks.append((
        _business_day_offset(src.value_date, 4), victim.net_paise,
        "CHARGEBACK-REVERSAL-" + src.utr,
    ))

    return plan


def _rebuild_with_gross(p: PlannedPayment, gross: int) -> PlannedPayment:
    fee, gst, net = net_for(gross)
    return PlannedPayment(p.payment_id, p.order_id, gross, fee, gst, net,
                          p.customer_name, p.order_date)


def _equalise_nets(a: PlannedBatch, b: PlannedBatch) -> None:
    """Adjust batch ``b``'s last payment so both batches share a net total."""
    target = sum(p.net_paise for p in a.payments)
    others = sum(p.net_paise for p in b.payments[:-1])
    needed_net = target - others
    if needed_net <= 0:  # pragma: no cover - guard; not expected at our volumes
        return
    # Invert net -> gross. Search a small window around the linear estimate
    # for an exact integer solution (rounding makes this non-analytic).
    est = round(needed_net / (1 - 0.02 - 0.02 * 0.18))
    for gross in range(max(1, est - 60), est + 60):
        if net_for(gross)[2] == needed_net:
            b.payments[-1] = _rebuild_with_gross(b.payments[-1], gross)
            return


# --- Emission -------------------------------------------------------------
# Everything below derives rows from the Plan. No case decisions are made
# here; if a label were assigned at this point it would already be suspect.

def _csv_write(path, header, rows):
    """Write CSV with explicit LF newlines so output is byte-identical on
    every OS. csv.writer defaults to CRLF, which would make the determinism
    diff platform-dependent."""
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def emit(plan: Plan, out_dir) -> dict:
    """Derive orders.csv, settlements.csv, bank.csv and truth.json from a Plan."""
    import json
    from datetime import datetime
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -- orders.csv --------------------------------------------------------
    # A refund debit narrates the order it reverses; those orders are marked
    # refunded rather than paid. Parsed from the known template prefix, not by
    # splitting on the last dash, so an order id containing a dash is safe.
    refund_prefix = "NEFT-RAZORPAY-REFUND-"
    refunded_orders = {
        n[len(refund_prefix):] for _, _, n in plan.refund_debits
        if n.startswith(refund_prefix)
    }
    order_rows = []
    for batch in plan.batches:
        for p in batch.payments:
            status = "refunded" if p.order_id in refunded_orders else "paid"
            # invoice_paise == the gross the customer was billed.
            order_rows.append([p.order_id, p.customer_name, p.gross_paise,
                               p.order_date.isoformat(), status])
    for oid, name, amount, created in plan.unpaid_orders:
        order_rows.append([oid, name, amount, created.isoformat(), "unpaid"])
    order_rows.sort(key=lambda r: r[0])
    _csv_write(out / "orders.csv",
               ["order_id", "customer_name", "invoice_paise", "created_at", "status"],
               order_rows)

    # -- settlements.csv ---------------------------------------------------
    pay_rows = []
    for batch in plan.batches:
        for p in batch.payments:
            # missing_utr batches carry no reference anywhere, including here.
            utr = "" if batch.case is Case.MISSING_UTR else batch.utr
            pay_rows.append([p.payment_id, p.order_id, p.gross_paise,
                             p.fee_paise, p.gst_paise, p.net_paise,
                             batch.settlement_id, batch.settled_at.isoformat(), utr])
    pay_rows.sort(key=lambda r: r[0])
    _csv_write(out / "settlements.csv",
               ["payment_id", "order_id", "gross_paise", "fee_paise", "gst_paise",
                "net_paise", "settlement_id", "settled_at", "utr"], pay_rows)

    # -- bank.csv ----------------------------------------------------------
    # Collect every row, sort by date, THEN compute the running balance so it
    # is arithmetically consistent down the statement as the validator demands.
    pending: list[tuple[date, int, int, str, Case, str]] = []
    for batch in plan.batches:
        narration = NARRATION_TEMPLATES[batch.template_idx].format(
            utr=batch.narration_utr) if batch.narration_utr else NO_REF_NARRATION
        pending.append((batch.value_date, batch.credit_paise(), 0, narration,
                        batch.case, batch.settlement_id))
    for value_date, narration, amount in plan.foreign_credits:
        pending.append((value_date, amount, 0, narration, Case.FOREIGN_CREDIT, ""))
    for value_date, amount, narration in plan.refund_debits:
        pending.append((value_date, 0, amount, narration, Case.REFUND_DEBIT, ""))
    for value_date, amount, narration in plan.chargebacks:
        pending.append((value_date, 0, amount, narration, Case.CHARGEBACK_REVERSAL, ""))

    # Stable sort: date, then credit-before-debit, then narration.
    pending.sort(key=lambda r: (r[0], r[2] > 0, r[3]))

    OPENING_BALANCE = 5_000_000  # Rs 50,000 opening float
    balance = OPENING_BALANCE
    bank_rows = []
    txn_of_batch: dict[str, str] = {}
    txn_case: list[tuple[str, Case]] = []
    for i, (value_date, credit, debit, narration, case, stl) in enumerate(pending, start=1):
        txn_id = "BNK%04d" % i
        balance += credit - debit
        bank_rows.append([txn_id, value_date.isoformat(), narration,
                          credit, debit, balance])
        if stl:
            txn_of_batch[stl] = txn_id
        txn_case.append((txn_id, case))
    _csv_write(out / "bank.csv",
               ["txn_id", "value_date", "narration", "credit_paise",
                "debit_paise", "balance_paise"], bank_rows)

    # -- truth.json --------------------------------------------------------
    matches = []
    for batch in plan.batches:
        matches.append({
            "bank_txn_id": txn_of_batch[batch.settlement_id],
            "payment_ids": sorted(p.payment_id for p in batch.claimed_payments()),
            "settlement_id": batch.settlement_id,
            "case": batch.case.value,
            "credit_paise": batch.credit_paise(),
            "withheld_payment_id": batch.withheld_payment_id,
            "drift_paise": batch.drift_paise,
        })
    matches.sort(key=lambda m: m["bank_txn_id"])

    unmatchable = [
        {"bank_txn_id": t, "case": c.value}
        for t, c in txn_case if c is Case.FOREIGN_CREDIT
    ]
    # Debit rows get an explicit home in truth so they are scored rather than
    # silently ignored (they can never appear as a credit match).
    debit_rows = [
        {"bank_txn_id": t, "case": c.value}
        for t, c in txn_case
        if c in (Case.REFUND_DEBIT, Case.CHARGEBACK_REVERSAL)
    ]

    truth = {
        "seed": plan.seed,
        "hard_ratio": plan.hard_ratio,
        "generated_at": datetime(2026, 1, 1).isoformat(),  # fixed: determinism
        "opening_balance_paise": OPENING_BALANCE,
        "counts": {
            "orders": len(order_rows), "payments": len(pay_rows),
            "settlements": len(plan.batches), "bank_rows": len(bank_rows),
        },
        "case_counts": {
            c: sum(1 for b in plan.batches if b.case.value == c)
            for c in sorted({b.case.value for b in plan.batches})
        },
        "matches": matches,
        "unmatchable_bank": unmatchable,
        "debit_rows": debit_rows,
        "unpaid_orders": [{"order_id": o, "case": "unpaid_order"}
                          for o, _, _, _ in plan.unpaid_orders],
    }
    with open(out / "truth.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(truth, fh, indent=2, sort_keys=False)
        fh.write("\n")

    return {"orders": len(order_rows), "payments": len(pay_rows),
            "bank_rows": len(bank_rows), "batches": len(plan.batches)}


def generate(seed: int, hard_ratio: float, out_dir,
             n_batches: int = DEFAULT_BATCHES) -> dict:
    """Build the truth plan, then derive all four files from it."""
    return emit(build_plan(seed, hard_ratio, n_batches), out_dir)

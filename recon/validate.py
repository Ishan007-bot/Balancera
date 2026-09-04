"""Ground-truth self-consistency checks.

Phase 1 is the most important phase in the project: every downstream metric is
measured against truth.json, so a subtly wrong truth file produces confident,
meaningless numbers. This module exists to make that failure mode impossible
to miss -- it re-derives every claim in truth.json from the CSVs and refuses
to pass if any of them disagree.

Every check returns a list of human-readable failure strings. An empty list
means the check passed.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import date
from pathlib import Path

# The reference-extraction regex. Defined here (not in the matcher) because
# validate must prove no foreign_credit narration can match it -- and it must
# be the *same* pattern the matcher will use in Phase 2.
REFERENCE_RE = re.compile(r"\b(?:UTR)?(\d{7,})\b|\b(UTR\d+)\b")


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def extract_reference(narration: str) -> str | None:
    """Pull a candidate payment reference out of bank narration text."""
    m = REFERENCE_RE.search(narration)
    if not m:
        return None
    return m.group(1) or m.group(2)


def validate(data_dir) -> list[str]:
    """Run every consistency check. Returns a list of failures (empty == pass)."""
    d = Path(data_dir)
    failures: list[str] = []

    for name in ("orders.csv", "settlements.csv", "bank.csv", "truth.json"):
        if not (d / name).exists():
            failures.append("MISSING FILE: %s" % name)
    if failures:
        return failures

    orders = _read_csv(d / "orders.csv")
    payments = _read_csv(d / "settlements.csv")
    bank = _read_csv(d / "bank.csv")
    truth = json.loads((d / "truth.json").read_text(encoding="utf-8"))

    pay_by_id = {p["payment_id"]: p for p in payments}
    bank_by_id = {b["txn_id"]: b for b in bank}
    order_ids = {o["order_id"] for o in orders}

    # 1. Per-payment arithmetic identity: gross - fee - gst == net.
    for p in payments:
        g, f, gs, n = (int(p["gross_paise"]), int(p["fee_paise"]),
                       int(p["gst_paise"]), int(p["net_paise"]))
        if g - f - gs != n:
            failures.append(
                "PAYMENT IDENTITY: %s gross=%d fee=%d gst=%d net=%d (expected net=%d)"
                % (p["payment_id"], g, f, gs, n, g - f - gs))

    # 2. Referential integrity: every id in truth must exist in the CSVs.
    for m in truth["matches"]:
        if m["bank_txn_id"] not in bank_by_id:
            failures.append("TRUTH REF: unknown bank_txn_id %s" % m["bank_txn_id"])
        for pid in m["payment_ids"]:
            if pid not in pay_by_id:
                failures.append("TRUTH REF: unknown payment_id %s in match %s"
                                % (pid, m["bank_txn_id"]))
    for u in truth["unmatchable_bank"] + truth.get("debit_rows", []):
        if u["bank_txn_id"] not in bank_by_id:
            failures.append("TRUTH REF: unknown unmatchable/debit txn %s"
                            % u["bank_txn_id"])
    for u in truth["unpaid_orders"]:
        if u["order_id"] not in order_ids:
            failures.append("TRUTH REF: unknown unpaid order %s" % u["order_id"])

    # 3. No payment may appear in more than one truth match. A payment claimed
    #    twice would make precision unmeasurable.
    seen: dict[str, str] = {}
    for m in truth["matches"]:
        for pid in m["payment_ids"]:
            if pid in seen:
                failures.append("DOUBLE CLAIM: %s in both %s and %s"
                                % (pid, seen[pid], m["bank_txn_id"]))
            seen[pid] = m["bank_txn_id"]

    # 4. Credit amounts must equal the sum of the claimed payments' net,
    #    plus the declared drift. clean_batch allows no drift at all.
    for m in truth["matches"]:
        txn = bank_by_id.get(m["bank_txn_id"])
        if txn is None:
            continue
        expected = sum(int(pay_by_id[p]["net_paise"]) for p in m["payment_ids"]
                       if p in pay_by_id)
        drift = m.get("drift_paise", 0)
        actual = int(txn["credit_paise"])
        if actual != expected + drift:
            failures.append(
                "CREDIT SUM (%s): %s credit=%d but payments sum=%d drift=%d"
                % (m["case"], m["bank_txn_id"], actual, expected, drift))
        if m["case"] == "clean_batch" and drift != 0:
            failures.append("CLEAN BATCH DRIFT: %s carries drift=%d"
                            % (m["bank_txn_id"], drift))

    # 5. partial_batch: credit == group total minus exactly the withheld
    #    payment, and the withheld net must be UNIQUE within its group.
    #    Without uniqueness two different subsets sum to the credit, the
    #    matcher correctly reports ambiguity, and truth becomes unmatchable.
    by_settlement: dict[str, list[dict]] = {}
    for p in payments:
        by_settlement.setdefault(p["settlement_id"], []).append(p)
    for m in truth["matches"]:
        if m["case"] != "partial_batch":
            continue
        withheld = m.get("withheld_payment_id")
        group = by_settlement.get(m["settlement_id"], [])
        if withheld is None:
            failures.append("PARTIAL BATCH: %s has no withheld_payment_id"
                            % m["bank_txn_id"])
            continue
        if withheld in m["payment_ids"]:
            failures.append("PARTIAL BATCH: withheld %s is also claimed in %s"
                            % (withheld, m["bank_txn_id"]))
        group_total = sum(int(p["net_paise"]) for p in group)
        w_net = int(pay_by_id[withheld]["net_paise"])
        credit = int(bank_by_id[m["bank_txn_id"]]["credit_paise"])
        if credit != group_total - w_net:
            failures.append(
                "PARTIAL BATCH: %s credit=%d != group %d - withheld %d"
                % (m["bank_txn_id"], credit, group_total, w_net))
        nets = [int(p["net_paise"]) for p in group]
        if nets.count(w_net) != 1:
            failures.append(
                "PARTIAL BATCH AMBIGUITY: withheld net %d appears %d times in %s"
                % (w_net, nets.count(w_net), m["settlement_id"]))

    # 6. Bank balances must be cumulative and arithmetically consistent.
    opening = truth.get("opening_balance_paise", 0)
    running = opening
    for row in bank:
        running += int(row["credit_paise"]) - int(row["debit_paise"])
        if running != int(row["balance_paise"]):
            failures.append("BALANCE: %s expected running %d, file says %s"
                            % (row["txn_id"], running, row["balance_paise"]))
            break  # one report is enough; everything after would cascade

    # 7. No foreign_credit narration may match the reference regex. If one
    #    did, stage 1 could "successfully" match a transaction that
    #    corresponds to nothing -- the exact failure mode we exist to prevent.
    for u in truth["unmatchable_bank"]:
        if u["case"] != "foreign_credit":
            continue
        narration = bank_by_id[u["bank_txn_id"]]["narration"]
        ref = extract_reference(narration)
        if ref is not None:
            failures.append(
                "FOREIGN CREDIT REGEX: %s narration %r yields reference %r"
                % (u["bank_txn_id"], narration, ref))

    # 8. Settlement-to-credit lag must actually match each case. Stage 2's
    #    date window is [settled_at, settled_at + N days], so the lag that
    #    matters is settled_at -> value_date. weekend_delay must exceed a
    #    naive 2-day window or the case tests nothing; every other batch must
    #    sit inside it, or the "hard" cases are indistinguishable from clean.
    for m in truth["matches"]:
        if not m["payment_ids"]:
            continue
        pid = m["payment_ids"][0]
        if pid not in pay_by_id or m["bank_txn_id"] not in bank_by_id:
            continue
        settled = date.fromisoformat(pay_by_id[pid]["settled_at"])
        value = date.fromisoformat(bank_by_id[m["bank_txn_id"]]["value_date"])
        lag = (value - settled).days
        if lag < 0:
            failures.append("DATE ORDER: %s value_date %s precedes settled_at %s"
                            % (m["bank_txn_id"], value, settled))
        elif m["case"] == "weekend_delay":
            if lag <= 2:
                failures.append(
                    "WEEKEND DELAY INERT: %s lag=%dd, does not exceed a naive "
                    "2-day window (case exercises nothing)"
                    % (m["bank_txn_id"], lag))
        elif lag > 2:
            failures.append("UNEXPECTED LAG: %s (%s) lag=%dd exceeds T+2"
                            % (m["bank_txn_id"], m["case"], lag))

    # 9. Case counts must match what was requested.
    declared = truth.get("case_counts", {})
    actual_counts: dict[str, int] = {}
    for m in truth["matches"]:
        actual_counts[m["case"]] = actual_counts.get(m["case"], 0) + 1
    for case, n in declared.items():
        if actual_counts.get(case, 0) != n:
            failures.append("CASE COUNT: %s declared %d, found %d"
                            % (case, n, actual_counts.get(case, 0)))

    # 10. Every bank row must be accounted for exactly once in truth.
    claimed_txns = ({m["bank_txn_id"] for m in truth["matches"]}
                    | {u["bank_txn_id"] for u in truth["unmatchable_bank"]}
                    | {u["bank_txn_id"] for u in truth.get("debit_rows", [])})
    for txn_id in bank_by_id:
        if txn_id not in claimed_txns:
            failures.append("ORPHAN BANK ROW: %s appears in no truth section"
                            % txn_id)

    # 11. Unpaid orders must genuinely have no payment.
    paid_order_ids = {p["order_id"] for p in payments}
    for u in truth["unpaid_orders"]:
        if u["order_id"] in paid_order_ids:
            failures.append("UNPAID ORDER: %s actually has a payment"
                            % u["order_id"])

    return failures

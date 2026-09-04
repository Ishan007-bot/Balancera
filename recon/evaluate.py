"""Scoring against ground truth.

Definitions are implemented exactly as specified, with two corrections agreed
during planning:

* **Auto-match rate counts only CORRECT matches.** "matched / matchable" left
  open whether a wrong match counts as matched -- under that reading a matcher
  that forces a guess on every row scores 100%. The numerator here is true
  positives.
* **Forced-match errors are a strict subset of false positives.** Both are
  reported, and the report states the overlap so the table cannot be read as
  double-counting.

A match is the pair ``(bank_txn_id, frozenset(payment_ids))``. A true positive
requires the payment set to equal truth's set exactly -- overlapping-but-not-
equal is a partial match, which counts against strict precision and is also
reported separately because hiding it would be dishonest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ingest import Dataset, Truth
from .models import Match


@dataclass
class Metrics:
    # Credit-side matching
    true_positives: int = 0
    false_positives: int = 0
    partial_matches: int = 0
    forced_match_errors: int = 0
    total_truth_matches: int = 0
    matchable_credits: int = 0
    proposals: int = 0

    # Order-side and non-credit rows: these exist so that every generated
    # record contributes to some reported number instead of vanishing.
    unpaid_orders_total: int = 0
    unpaid_orders_detected: int = 0
    foreign_credits_total: int = 0
    foreign_credits_left_unmatched: int = 0
    debit_rows_total: int = 0
    debit_rows_left_unmatched: int = 0

    detail: dict = field(default_factory=dict)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        return (self.true_positives / self.total_truth_matches
                if self.total_truth_matches else 0.0)

    @property
    def auto_match_rate(self) -> float:
        """Correctly matched credits over matchable credits."""
        return (self.true_positives / self.matchable_credits
                if self.matchable_credits else 0.0)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def unpaid_order_recall(self) -> float:
        return (self.unpaid_orders_detected / self.unpaid_orders_total
                if self.unpaid_orders_total else 0.0)

    def as_dict(self) -> dict:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "partial_matches": self.partial_matches,
            "forced_match_errors": self.forced_match_errors,
            "total_truth_matches": self.total_truth_matches,
            "matchable_credits": self.matchable_credits,
            "proposals": self.proposals,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "auto_match_rate": round(self.auto_match_rate, 4),
            "f1": round(self.f1, 4),
            "unpaid_orders_total": self.unpaid_orders_total,
            "unpaid_orders_detected": self.unpaid_orders_detected,
            "unpaid_order_recall": round(self.unpaid_order_recall, 4),
            "foreign_credits_total": self.foreign_credits_total,
            "foreign_credits_left_unmatched": self.foreign_credits_left_unmatched,
            "debit_rows_total": self.debit_rows_total,
            "debit_rows_left_unmatched": self.debit_rows_left_unmatched,
            "detail": self.detail,
        }


def evaluate(matches: list[Match], truth: Truth, ds: Dataset,
             detected_unpaid: set[str] | None = None) -> Metrics:
    """Score a set of proposed matches against ground truth."""
    m = Metrics()
    truth_by_txn = truth.by_txn
    m.total_truth_matches = len(truth.matches)
    m.matchable_credits = len(truth.matchable_txn_ids)
    m.proposals = len(matches)

    tp_ids, fp_detail, partial_detail, forced_detail = [], [], [], []

    for match in matches:
        expected = truth_by_txn.get(match.bank_txn_id)

        # A match proposed against a row that should match nothing is the
        # single worst outcome the system can produce.
        if expected is None:
            m.false_positives += 1
            if (match.bank_txn_id in truth.unmatchable_bank
                    or match.bank_txn_id in truth.debit_rows):
                m.forced_match_errors += 1
                case = (truth.unmatchable_bank.get(match.bank_txn_id)
                        or truth.debit_rows.get(match.bank_txn_id))
                forced_detail.append({
                    "bank_txn_id": match.bank_txn_id,
                    "case": str(case),
                    "stage": match.stage,
                    "proposed": sorted(match.payment_ids),
                })
            else:
                fp_detail.append({
                    "bank_txn_id": match.bank_txn_id,
                    "stage": match.stage,
                    "reason": "no truth match exists for this transaction",
                })
            continue

        if match.payment_ids == expected.payment_ids:
            m.true_positives += 1
            tp_ids.append(match.bank_txn_id)
            continue

        # Wrong set. Strictly a false positive; also reported as partial when
        # it overlaps, because "nearly right" and "entirely wrong" are
        # different diagnoses even though they score the same.
        m.false_positives += 1
        overlap = match.payment_ids & expected.payment_ids
        if overlap:
            m.partial_matches += 1
            partial_detail.append({
                "bank_txn_id": match.bank_txn_id,
                "stage": match.stage,
                "case": str(expected.case),
                "proposed": sorted(match.payment_ids),
                "expected": sorted(expected.payment_ids),
                "missing": sorted(expected.payment_ids - match.payment_ids),
                "extra": sorted(match.payment_ids - expected.payment_ids),
            })
        else:
            fp_detail.append({
                "bank_txn_id": match.bank_txn_id,
                "stage": match.stage,
                "case": str(expected.case),
                "reason": "payment set disjoint from truth",
            })

    # Order-side: unpaid orders can never appear as a credit match, so they
    # need their own metric or the three of them affect nothing.
    m.unpaid_orders_total = len(truth.unpaid_orders)
    detected = detected_unpaid if detected_unpaid is not None else set()
    m.unpaid_orders_detected = len(detected & set(truth.unpaid_orders))

    matched_txns = {mt.bank_txn_id for mt in matches}
    m.foreign_credits_total = len(truth.unmatchable_bank)
    m.foreign_credits_left_unmatched = sum(
        1 for t in truth.unmatchable_bank if t not in matched_txns)
    m.debit_rows_total = len(truth.debit_rows)
    m.debit_rows_left_unmatched = sum(
        1 for t in truth.debit_rows if t not in matched_txns)

    m.detail = {
        "true_positive_txns": sorted(tp_ids),
        "false_positives": fp_detail,
        "partial_matches": partial_detail,
        "forced_matches": forced_detail,
        "missed_txns": sorted(truth.matchable_txn_ids - matched_txns),
    }
    return m


def per_case_breakdown(matches: list[Match], truth: Truth) -> dict:
    """Match rate per generator case -- which hard cases actually got solved.

    More informative than a single headline number: it shows whether the
    system is strong overall but blind to one specific failure mode.
    """
    truth_by_txn = truth.by_txn
    proposed = {mt.bank_txn_id: mt.payment_ids for mt in matches}
    out: dict[str, dict] = {}
    for tm in truth.matches:
        row = out.setdefault(str(tm.case), {"total": 0, "correct": 0, "wrong": 0})
        row["total"] += 1
        got = proposed.get(tm.bank_txn_id)
        if got is None:
            continue
        if got == tm.payment_ids:
            row["correct"] += 1
        else:
            row["wrong"] += 1
    for row in out.values():
        row["rate"] = round(row["correct"] / row["total"], 4) if row["total"] else 0.0
    return dict(sorted(out.items()))


def format_metrics_table(m: Metrics, title: str = "Metrics") -> str:
    """Plain-text metrics table for the CLI."""
    lines = [
        title,
        "=" * len(title),
        "  Auto-match rate      %6.1f%%   (%d correct / %d matchable credits)"
        % (100 * m.auto_match_rate, m.true_positives, m.matchable_credits),
        "  Precision            %6.1f%%   (%d TP / %d proposals)"
        % (100 * m.precision, m.true_positives, m.true_positives + m.false_positives),
        "  Recall               %6.1f%%   (%d TP / %d truth matches)"
        % (100 * m.recall, m.true_positives, m.total_truth_matches),
        "  F1                   %6.1f%%" % (100 * m.f1),
        "",
        "  True positives       %6d" % m.true_positives,
        "  False positives      %6d" % m.false_positives,
        "  ...of which partial  %6d   (overlapping but not exact)" % m.partial_matches,
        "  Forced-match errors  %6d   %s"
        % (m.forced_match_errors,
           "<-- MUST BE ZERO" if m.forced_match_errors else "(subset of FP above)"),
        "",
        "  Unpaid orders        %6d / %d detected" % (m.unpaid_orders_detected,
                                                      m.unpaid_orders_total),
        "  Foreign credits      %6d / %d correctly left unmatched"
        % (m.foreign_credits_left_unmatched, m.foreign_credits_total),
        "  Debit rows           %6d / %d correctly left unmatched"
        % (m.debit_rows_left_unmatched, m.debit_rows_total),
    ]
    return "\n".join(lines)

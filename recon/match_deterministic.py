"""Deterministic matching, stages 1-3. No LLM anywhere in this module.

These three stages produce the **baseline** the entire submission is measured
against: the headline number is the delta between what this achieves alone and
what it achieves with the LLM layer on top. That comparison is only honest if
the baseline is recorded before the LLM is ever called.

The governing rule, in all three stages: **when the evidence is ambiguous, do
not pick.** A tie left unresolved becomes an exception a human reviews. A tie
guessed wrongly becomes a corrupted ledger nobody notices. Every stage here
would score better on naive "match rate" by guessing; none of them do.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import timedelta

from .ingest import Dataset, parse_date
from .models import Match

# Shared with validate.py by design: the validator proves no foreign_credit
# narration can match this exact pattern, which is only a real guarantee if
# the matcher uses the same one.
REFERENCE_RE = re.compile(r"\b(?:UTR)?(\d{7,})\b|\b(UTR\d+)\b")

# Defaults, overridable per run.
DATE_WINDOW_DAYS = 3   # value_date within [settled_at, settled_at + N]
AMOUNT_TOLERANCE = 2   # paise, absorbs fee-rounding drift
MAX_SUBSET_GROUP = 15  # beyond this, skip to the LLM stage


@dataclass
class MatchResult:
    """Everything one deterministic run produced, including what it refused."""

    matches: list[Match] = field(default_factory=list)
    #: credit txn_id -> why it could not be resolved deterministically
    unresolved: dict[str, str] = field(default_factory=dict)
    #: credit txn_id -> the competing candidates, for the LLM stage and report
    ambiguous: dict[str, list] = field(default_factory=dict)
    #: per-stage counts, for the ablation table
    stage_counts: dict[str, int] = field(default_factory=dict)

    @property
    def claimed_payment_ids(self) -> set[str]:
        out: set[str] = set()
        for m in self.matches:
            out |= set(m.payment_ids)
        return out

    @property
    def matched_txn_ids(self) -> set[str]:
        return {m.bank_txn_id for m in self.matches}


def canonical_ref(ref: str) -> str:
    """Reduce a reference to its digits.

    Narration and the settlement file disagree about the ``UTR`` prefix --
    ``UTR8973915`` in one, ``8973915`` in the other, depending on the
    template. Comparing the digits is what makes a reference match a
    reference rather than a formatting coincidence.
    """
    return "".join(ch for ch in ref if ch.isdigit())


def extract_reference(narration: str) -> str | None:
    """Pull a candidate payment reference out of bank narration text."""
    m = REFERENCE_RE.search(narration)
    if not m:
        return None
    return m.group(1) or m.group(2)


def _within_window(settled_at: str, value_date: str, window_days: int) -> bool:
    """Is the credit's value_date inside [settled_at, settled_at + window]?

    Deliberately asymmetric: money cannot reach the bank before the gateway
    settles it, so a credit dated earlier than settlement is not a late
    settlement -- it is a different transaction.
    """
    settled = parse_date(settled_at)
    value = parse_date(value_date)
    return settled <= value <= settled + timedelta(days=window_days)


# --- Stage 1: reference match ---------------------------------------------

def stage1_reference(ds: Dataset, result: MatchResult,
                     tolerance: int = AMOUNT_TOLERANCE) -> None:
    """Match on a reference extracted from narration, then verify the amount.

    A reference hit whose amount disagrees is NOT a match -- it is a data
    problem worth surfacing. Accepting it because the UTR lined up is exactly
    the failure mode that makes reconciliation software dangerous.
    """
    matched = 0
    for txn in ds.credits:
        if txn.txn_id in result.matched_txn_ids:
            continue
        ref = extract_reference(txn.narration)
        if ref is None:
            continue

        group = ds.payments_by_utr.get(canonical_ref(ref))
        if not group:
            # The reference may be a settlement id rather than a UTR.
            group = ds.payments_by_settlement.get(ref)
        if not group:
            continue

        available = [p for p in group
                     if p.payment_id not in result.claimed_payment_ids]
        if not available:
            continue

        total = sum(p.net_paise for p in available)
        delta = abs(total - txn.credit_paise)
        if delta > tolerance:
            result.unresolved[txn.txn_id] = (
                "reference %s matched settlement %s, but amounts disagree: "
                "credit %d vs payments %d (off by %d paise)"
                % (ref, available[0].settlement_id, txn.credit_paise, total, delta))
            continue

        if not _within_window(available[0].settled_at, txn.value_date,
                              DATE_WINDOW_DAYS):
            result.unresolved[txn.txn_id] = (
                "reference %s matched, amount agrees, but value_date %s is "
                "outside the settlement window from %s"
                % (ref, txn.value_date, available[0].settled_at))
            continue

        result.matches.append(Match(
            bank_txn_id=txn.txn_id,
            payment_ids=frozenset(p.payment_id for p in available),
            stage="reference",
            confidence=1.0,
            reasoning="narration reference %s resolves to settlement %s; "
                      "%d payments sum to %d against credit %d"
                      % (ref, available[0].settlement_id, len(available),
                         total, txn.credit_paise),
        ))
        matched += 1
    result.stage_counts["reference"] = matched


# --- Stage 2: group sum ---------------------------------------------------

def stage2_group_sum(ds: Dataset, result: MatchResult,
                     window_days: int = DATE_WINDOW_DAYS,
                     tolerance: int = AMOUNT_TOLERANCE) -> None:
    """Match whole settlement groups by summed net amount within a date window.

    Resolves the batches whose reference was mangled or absent. If two groups
    tie for one credit, both are recorded as ambiguous and NEITHER is chosen.
    """
    matched = 0
    for txn in ds.credits:
        if txn.txn_id in result.matched_txn_ids:
            continue

        candidates = []
        for stl, payments in ds.payments_by_settlement.items():
            available = [p for p in payments
                         if p.payment_id not in result.claimed_payment_ids]
            if not available:
                continue
            total = sum(p.net_paise for p in available)
            if abs(total - txn.credit_paise) > tolerance:
                continue
            if not _within_window(available[0].settled_at, txn.value_date,
                                  window_days):
                continue
            candidates.append((stl, available, total))

        if not candidates:
            continue

        if len(candidates) > 1:
            # A genuine tie. Guessing here is the exact failure this project
            # exists to prevent, so record it and move on.
            result.ambiguous[txn.txn_id] = [
                {"settlement_id": stl,
                 "payment_ids": sorted(p.payment_id for p in avail),
                 "total_paise": total}
                for stl, avail, total in candidates
            ]
            result.unresolved[txn.txn_id] = (
                "%d settlement groups match credit %d within %d paise on the "
                "same date window (%s) -- genuine tie, not guessing"
                % (len(candidates), txn.credit_paise, tolerance,
                   ", ".join(c[0] for c in candidates)))
            continue

        stl, available, total = candidates[0]
        result.matches.append(Match(
            bank_txn_id=txn.txn_id,
            payment_ids=frozenset(p.payment_id for p in available),
            stage="group_sum",
            confidence=1.0,
            reasoning="settlement %s: %d payments sum to %d against credit %d "
                      "(delta %d paise), settled %s, credited %s"
                      % (stl, len(available), total, txn.credit_paise,
                         total - txn.credit_paise, available[0].settled_at,
                         txn.value_date),
        ))
        matched += 1
    result.stage_counts["group_sum"] = matched


# --- Stage 3: bounded subset sum ------------------------------------------

def find_subsets(amounts: list[int], target: int, tolerance: int,
                 limit: int = 2) -> list[tuple[int, ...]]:
    """Find index subsets summing to ``target`` within ``tolerance``.

    Meet-in-the-middle: split the list, enumerate both halves, then match one
    against the other. At n<=15 a naive powerset would also run instantly --
    the reason for MITM is a bounded worst case rather than raw speed, which
    matters because group size is data-dependent and not ours to control.

    Returns up to ``limit`` solutions. Finding two is enough to declare
    ambiguity; enumerating all of them would waste time we cannot use.
    """
    n = len(amounts)
    if n == 0:
        return []

    half = n // 2
    left_idx = list(range(half))
    right_idx = list(range(half, n))

    def enumerate_half(indices):
        sums = [(0, ())]
        for i in indices:
            sums += [(s + amounts[i], subset + (i,)) for s, subset in sums]
        return sums

    left = enumerate_half(left_idx)
    right = sorted(enumerate_half(right_idx))
    right_sums = [s for s, _ in right]

    import bisect
    found: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for lsum, lsub in left:
        lo = bisect.bisect_left(right_sums, target - lsum - tolerance)
        hi = bisect.bisect_right(right_sums, target - lsum + tolerance)
        for j in range(lo, hi):
            rsum, rsub = right[j]
            if abs(lsum + rsum - target) > tolerance:
                continue
            combined = tuple(sorted(lsub + rsub))
            if not combined or combined in seen:
                continue  # the empty subset is never a match
            seen.add(combined)
            found.append(combined)
            if len(found) >= limit:
                return found
    return found


def stage3_subset_sum(ds: Dataset, result: MatchResult,
                      window_days: int = DATE_WINDOW_DAYS,
                      tolerance: int = AMOUNT_TOLERANCE,
                      max_group: int = MAX_SUBSET_GROUP) -> None:
    """Resolve partial batches: a credit covering all but some of a group.

    Bounded hard -- only within a single settlement group, and only when that
    group is small enough that the search is provably cheap. Larger groups are
    left for the LLM stage rather than allowed to blow up here.
    """
    matched = 0
    for txn in ds.credits:
        if txn.txn_id in result.matched_txn_ids:
            continue

        best: list[tuple[str, list, tuple[int, ...]]] = []
        for stl, payments in ds.payments_by_settlement.items():
            available = [p for p in payments
                         if p.payment_id not in result.claimed_payment_ids]
            if not available or len(available) > max_group:
                continue
            if not _within_window(available[0].settled_at, txn.value_date,
                                  window_days):
                continue
            # A subset can only be smaller than the whole; if the group total
            # is already short of the credit, no subset can reach it.
            if sum(p.net_paise for p in available) < txn.credit_paise - tolerance:
                continue

            subsets = find_subsets([p.net_paise for p in available],
                                   txn.credit_paise, tolerance, limit=2)
            for sub in subsets:
                best.append((stl, [available[i] for i in sub], sub))
            if len(best) > 1:
                break

        if not best:
            continue

        if len(best) > 1:
            result.ambiguous[txn.txn_id] = [
                {"settlement_id": stl,
                 "payment_ids": sorted(p.payment_id for p in chosen),
                 "total_paise": sum(p.net_paise for p in chosen)}
                for stl, chosen, _ in best
            ]
            result.unresolved[txn.txn_id] = (
                "%d distinct payment subsets sum to credit %d within %d paise "
                "-- ambiguous, not guessing"
                % (len(best), txn.credit_paise, tolerance))
            continue

        stl, chosen, _ = best[0]
        total = sum(p.net_paise for p in chosen)
        group_size = len(ds.payments_by_settlement[stl])
        result.matches.append(Match(
            bank_txn_id=txn.txn_id,
            payment_ids=frozenset(p.payment_id for p in chosen),
            stage="subset_sum",
            confidence=1.0,
            reasoning="settlement %s: %d of %d payments sum to %d against "
                      "credit %d (delta %d paise) -- remainder appears withheld"
                      % (stl, len(chosen), group_size, total, txn.credit_paise,
                         total - txn.credit_paise),
        ))
        matched += 1
    result.stage_counts["subset_sum"] = matched


# --- Orchestration --------------------------------------------------------

def run_deterministic(ds: Dataset, stages: int = 3,
                      window_days: int = DATE_WINDOW_DAYS,
                      tolerance: int = AMOUNT_TOLERANCE) -> MatchResult:
    """Run stages 1..``stages``. ``stages`` drives the ablation table."""
    result = MatchResult()
    if stages >= 1:
        stage1_reference(ds, result, tolerance)
    if stages >= 2:
        stage2_group_sum(ds, result, window_days, tolerance)
    if stages >= 3:
        stage3_subset_sum(ds, result, window_days, tolerance)

    # Clear stale reasons: an earlier stage may have recorded why *it* could
    # not resolve a credit that a later stage then matched. Leaving those in
    # would put resolved transactions on the exception list.
    for txn_id in result.matched_txn_ids:
        result.unresolved.pop(txn_id, None)
        result.ambiguous.pop(txn_id, None)

    for txn in ds.credits:
        if txn.txn_id not in result.matched_txn_ids:
            result.unresolved.setdefault(
                txn.txn_id,
                "no settlement group or subset matches credit %d within %d "
                "paise in the date window" % (txn.credit_paise, tolerance))
    return result

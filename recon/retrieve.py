"""Deterministic candidate retrieval for the LLM stage.

The model never sees the full dataset. For each residual credit we
deterministically select the top-K plausible payments, which bounds three
things at once: token cost, latency, and -- most importantly -- the surface
area for hallucination. A model shown 190 payments can invent a combination
from anywhere; a model shown 20 cannot.

Retrieval is pure and seeded by nothing: same input, same candidates, every
time. That keeps the prompt hash stable so the cache actually hits.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from .ingest import Dataset, parse_date
from .models import BankTxn, Payment

DEFAULT_K = 20
#: Retrieval is deliberately wider than the matcher's window. The point is to
#: give the model a chance on cases the deterministic stages rejected, some of
#: which failed *because* of a strict window.
RETRIEVAL_WINDOW_DAYS = 6


@dataclass
class Candidate:
    payment: Payment
    settlement_id: str
    days_from_settlement: int
    reason: str  # why this payment was offered, shown to the model


@dataclass
class CandidateSet:
    """What the model is allowed to consider for one credit."""

    txn: BankTxn
    candidates: list[Candidate]
    settlement_totals: dict[str, int]  # settlement_id -> sum of offered nets

    @property
    def payment_ids(self) -> set[str]:
        return {c.payment.payment_id for c in self.candidates}


def retrieve(ds: Dataset, txn: BankTxn, claimed: set[str],
             k: int = DEFAULT_K,
             window_days: int = RETRIEVAL_WINDOW_DAYS) -> CandidateSet:
    """Select up to ``k`` plausible unclaimed payments for one credit.

    Ordering is by settlement-group plausibility, then by date proximity, so
    payments that belong together arrive together -- a model shown a coherent
    group can reason about it; one shown 20 unrelated payments cannot.
    """
    value_date = parse_date(txn.value_date)

    # Score whole settlement groups first: a bank credit settles a batch, so
    # the unit of plausibility is the group, not the individual payment.
    group_scores: list[tuple[float, str, list[Payment]]] = []
    for stl, payments in ds.payments_by_settlement.items():
        available = [p for p in payments if p.payment_id not in claimed]
        if not available:
            continue
        settled = parse_date(available[0].settled_at)
        lag = (value_date - settled).days
        if not (0 <= lag <= window_days):
            continue
        total = sum(p.net_paise for p in available)
        # Closeness of the group total to the credit, as a relative distance.
        amount_gap = abs(total - txn.credit_paise) / max(txn.credit_paise, 1)
        group_scores.append((amount_gap + lag * 0.01, stl, available))

    group_scores.sort(key=lambda row: (row[0], row[1]))

    candidates: list[Candidate] = []
    totals: dict[str, int] = {}
    for score, stl, payments in group_scores:
        if len(candidates) >= k:
            break
        settled = parse_date(payments[0].settled_at)
        lag = (value_date - settled).days
        group_total = sum(p.net_paise for p in payments)
        for p in sorted(payments, key=lambda x: x.payment_id):
            if len(candidates) >= k:
                break
            candidates.append(Candidate(
                payment=p, settlement_id=stl, days_from_settlement=lag,
                reason="settlement %s totals %d across %d unclaimed payments; "
                       "credit is %d" % (stl, group_total, len(payments),
                                         txn.credit_paise),
            ))
            totals[stl] = group_total

    return CandidateSet(txn=txn, candidates=candidates, settlement_totals=totals)

"""The verification gate.

This is the most important module in the repository. Every proposed match --
whoever proposed it -- is re-checked here from raw source data before it can
become a financial fact.

Two design rules, both deliberate and both load-bearing:

1. **This module knows nothing about LLMs.** It does not import ``match_llm``,
   it has no concept of a model, and it cannot tell a model's proposal from a
   deterministic stage's. It takes a payment set and an amount and decides
   whether the arithmetic holds. That independence is what makes it a gate
   rather than a rubber stamp -- a checker that shares assumptions with the
   thing it checks is not checking anything.

2. **Never trust arithmetic that arrives from outside.** The proposer may
   state a sum; we recompute it from ``payments_by_id``. A model's confident
   "these sum to 3215986" is treated as a claim to be tested, not a fact.

Every rejection carries the rule that failed and the actual numbers involved,
because the rejection log is a deliverable: it is the evidence that the safety
layer does real work rather than passing everything through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .ingest import Dataset, parse_date
from .money import format_paise

DEFAULT_CONFIDENCE_THRESHOLD = 0.7
#: Cross-settlement proposals are unusual and need stronger evidence.
CROSS_SETTLEMENT_CONFIDENCE = 0.85

#: Amount tolerance, in paise. Deliberately tight, and it costs us: on the
#: default dataset it rejects a proposal whose payment set is *correct*,
#: because the credit carries 5 paise of fee-rounding drift. That is the
#: intended behaviour. A 5-paise gap the system cannot explain is exactly what
#: an unnoticed short-settlement looks like, and a gate that waves through
#: unexplained differences is not a gate. The match becomes an exception a
#: human clears in seconds; the alternative is a system that quietly absorbs
#: discrepancies it should have raised.
DEFAULT_TOLERANCE = 2
DEFAULT_WINDOW_DAYS = 3


class Rule(str, Enum):
    """The rules a proposal must satisfy. Each maps to one rejection reason."""

    EMPTY_PROPOSAL = "empty_proposal"
    UNKNOWN_PAYMENT = "unknown_payment"
    ALREADY_CLAIMED = "already_claimed"
    AMOUNT_MISMATCH = "amount_mismatch"
    DATE_WINDOW = "date_window"
    CROSS_SETTLEMENT = "cross_settlement"
    LOW_CONFIDENCE = "low_confidence"
    CONFLICT = "conflict"
    NOT_A_CREDIT = "not_a_credit"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Verdict:
    """The outcome of verifying one proposal."""

    accepted: bool
    bank_txn_id: str
    payment_ids: frozenset[str]
    failed_rule: Rule | None = None
    reason: str = ""
    detail: dict = field(default_factory=dict)

    def as_log_record(self) -> dict:
        """A human-readable rejection record.

        Includes the actual amounts, not just a rule name -- the log has to be
        readable aloud: "the model proposed X, the verifier rejected it
        because the payments sum to Y against a credit of Z".
        """
        return {
            "bank_txn_id": self.bank_txn_id,
            "proposed_payment_ids": sorted(self.payment_ids),
            "accepted": self.accepted,
            "failed_rule": str(self.failed_rule) if self.failed_rule else None,
            "reason": self.reason,
            **self.detail,
        }


def verify(
    bank_txn_id: str,
    payment_ids,
    confidence: float,
    ds: Dataset,
    claimed: set[str],
    accepted_txns: set[str] | None = None,
    tolerance: int = DEFAULT_TOLERANCE,
    window_days: int = DEFAULT_WINDOW_DAYS,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Verdict:
    """Re-check a proposed match from source data. Pure function.

    Rules are evaluated in order of how cheap and how damning they are: an
    unknown payment id is a harder failure than low confidence, and knowing
    which rule fired first makes the rejection log readable.
    """
    ids = frozenset(payment_ids)
    accepted_txns = accepted_txns or set()

    def reject(rule: Rule, reason: str, **detail) -> Verdict:
        return Verdict(False, bank_txn_id, ids, rule, reason, detail)

    # Rule 0: the target must be a credit row that exists.
    txn = ds.bank_by_id.get(bank_txn_id)
    if txn is None:
        return reject(Rule.NOT_A_CREDIT,
                      "bank transaction %s does not exist" % bank_txn_id)
    if not txn.is_credit:
        return reject(
            Rule.NOT_A_CREDIT,
            "%s is a debit of %s, not a credit -- payments cannot settle "
            "against it" % (bank_txn_id, format_paise(txn.debit_paise)),
            debit_paise=txn.debit_paise)

    # Rule 1: an empty proposal is not a match.
    if not ids:
        return reject(Rule.EMPTY_PROPOSAL,
                      "no payments proposed for credit %s"
                      % format_paise(txn.credit_paise),
                      credit_paise=txn.credit_paise)

    # Rule 2: every proposed payment must actually exist. This is the rule
    # that catches a hallucinated identifier.
    unknown = sorted(pid for pid in ids if pid not in ds.payments_by_id)
    if unknown:
        return reject(
            Rule.UNKNOWN_PAYMENT,
            "proposed payment id(s) %s do not exist in settlements.csv"
            % ", ".join(unknown),
            unknown_payment_ids=unknown, credit_paise=txn.credit_paise)

    # Rule 3: a payment already settled against another credit is not
    # available. Accepting it would double-count the money.
    taken = sorted(pid for pid in ids if pid in claimed)
    if taken:
        return reject(
            Rule.ALREADY_CLAIMED,
            "payment(s) %s are already claimed by another match"
            % ", ".join(taken),
            already_claimed=taken, credit_paise=txn.credit_paise)

    # Rule 4: the arithmetic, recomputed from source. Never trust a stated sum.
    payments = [ds.payments_by_id[pid] for pid in sorted(ids)]
    total = sum(p.net_paise for p in payments)
    delta = total - txn.credit_paise
    if abs(delta) > tolerance:
        return reject(
            Rule.AMOUNT_MISMATCH,
            "%d proposed payments sum to %s but the credit is %s -- off by "
            "%s (tolerance %d paise)"
            % (len(payments), format_paise(total),
               format_paise(txn.credit_paise), format_paise(abs(delta)),
               tolerance),
            proposed_sum_paise=total, credit_paise=txn.credit_paise,
            delta_paise=delta, tolerance_paise=tolerance)

    # Rule 5: each payment must have settled inside the window before the
    # credit landed. Money cannot arrive before the gateway releases it.
    value_date = parse_date(txn.value_date)
    for p in payments:
        settled = parse_date(p.settled_at)
        lag = (value_date - settled).days
        if not (0 <= lag <= window_days):
            return reject(
                Rule.DATE_WINDOW,
                "payment %s settled %s but the credit is dated %s (%d days "
                "apart; allowed 0 to %d)"
                % (p.payment_id, p.settled_at, txn.value_date, lag, window_days),
                payment_id=p.payment_id, settled_at=p.settled_at,
                value_date=txn.value_date, lag_days=lag,
                window_days=window_days)

    # Rule 6: a credit normally settles one batch. Spanning several is
    # possible but unusual, so it needs stronger evidence rather than a ban.
    settlements = sorted({p.settlement_id for p in payments})
    if len(settlements) > 1 and confidence < CROSS_SETTLEMENT_CONFIDENCE:
        return reject(
            Rule.CROSS_SETTLEMENT,
            "proposal spans %d settlements (%s) at confidence %.2f; %.2f is "
            "required for a cross-settlement match"
            % (len(settlements), ", ".join(settlements), confidence,
               CROSS_SETTLEMENT_CONFIDENCE),
            settlement_ids=settlements, confidence=confidence,
            required_confidence=CROSS_SETTLEMENT_CONFIDENCE)

    # Rule 7: confidence floor.
    if confidence < confidence_threshold:
        return reject(
            Rule.LOW_CONFIDENCE,
            "confidence %.2f is below the acceptance threshold %.2f"
            % (confidence, confidence_threshold),
            confidence=confidence, threshold=confidence_threshold)

    # Rule 8: one credit, one match.
    if bank_txn_id in accepted_txns:
        return reject(
            Rule.CONFLICT,
            "credit %s already has an accepted match" % bank_txn_id)

    return Verdict(
        accepted=True, bank_txn_id=bank_txn_id, payment_ids=ids,
        reason="%d payments sum to %s against a credit of %s (delta %d paise); "
               "settlement %s; all within the %d-day window"
               % (len(payments), format_paise(total),
                  format_paise(txn.credit_paise), delta,
                  ", ".join(settlements), window_days),
        detail={"proposed_sum_paise": total, "credit_paise": txn.credit_paise,
                "delta_paise": delta, "settlement_ids": settlements,
                "confidence": confidence},
    )


@dataclass
class GateResult:
    """What the gate accepted, what it rejected, and why."""

    accepted: list[Verdict] = field(default_factory=list)
    rejected: list[Verdict] = field(default_factory=list)

    @property
    def rejections_by_rule(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for v in self.rejected:
            key = str(v.failed_rule)
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items()))

    def summary(self) -> dict:
        return {
            "proposals_verified": len(self.accepted) + len(self.rejected),
            "accepted": len(self.accepted),
            "rejected": len(self.rejected),
            "rejections_by_rule": self.rejections_by_rule,
        }


def verify_all(proposals, ds: Dataset, claimed: set[str],
               tolerance: int = DEFAULT_TOLERANCE,
               window_days: int = DEFAULT_WINDOW_DAYS,
               confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
               ) -> GateResult:
    """Verify a sequence of proposals, accumulating claims as they are accepted.

    ``proposals`` is any iterable of objects exposing ``bank_txn_id``,
    ``proposed_payment_ids`` and ``confidence`` -- deliberately structural, so
    this function has no dependency on where a proposal came from.
    """
    result = GateResult()
    working_claimed = set(claimed)
    accepted_txns: set[str] = set()

    for p in proposals:
        verdict = verify(
            bank_txn_id=p.bank_txn_id,
            payment_ids=p.proposed_payment_ids,
            confidence=p.confidence,
            ds=ds,
            claimed=working_claimed,
            accepted_txns=accepted_txns,
            tolerance=tolerance,
            window_days=window_days,
            confidence_threshold=confidence_threshold,
        )
        if verdict.accepted:
            result.accepted.append(verdict)
            # An accepted match consumes its payments, so a later proposal
            # cannot claim them again within the same run.
            working_claimed |= set(verdict.payment_ids)
            accepted_txns.add(verdict.bank_txn_id)
        else:
            result.rejected.append(verdict)
    return result

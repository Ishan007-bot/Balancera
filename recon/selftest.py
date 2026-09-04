"""Adversarial self-test of the verification gate.

A rejection log proves the gate rejected *something*, but a reviewer has to
take on trust that the rejections were not cherry-picked. This module removes
that doubt: it takes matches the deterministic stages already accepted,
corrupts them in specific, realistic ways, and shows the gate catching each
one -- live, reproducibly, on demand.

Every corruption models a real failure. A shifted date is a settlement
attributed to the wrong day. An altered amount is a short-settlement nobody
noticed. A duplicated payment is money counted twice. An invented id is a
model hallucination. These are the ways reconciliation software goes wrong in
production, and each one has to fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ingest import Dataset
from .models import BankTxn
from .money import format_paise
from .verify import Rule, verify


@dataclass
class Attack:
    """One injected corruption and what the gate should do about it."""

    name: str
    description: str
    bank_txn_id: str
    payment_ids: list[str]
    confidence: float
    expected_rule: Rule
    claimed: set[str]
    mutate_txn: BankTxn | None = None  # a doctored bank row, if the attack needs one


def build_attacks(ds: Dataset, matches, claimed: set[str]) -> list[Attack]:
    """Derive attacks from real accepted matches.

    Using genuine matches matters: corrupting something the system already got
    right proves the gate catches regressions in working code, not just
    obviously broken input.
    """
    usable = [m for m in matches if len(m.payment_ids) >= 2]
    if not usable:
        return []
    attacks: list[Attack] = []

    # 1. Amount tampering -- a short settlement, or a fee silently applied twice.
    m = usable[0]
    txn = ds.bank_by_id[m.bank_txn_id]
    doctored = BankTxn(
        txn_id=txn.txn_id, value_date=txn.value_date, narration=txn.narration,
        credit_paise=txn.credit_paise + 10_000,  # Rs 100 more than was paid
        debit_paise=0, balance_paise=txn.balance_paise)
    attacks.append(Attack(
        name="amount tampered",
        description="credit inflated by %s over the payments it settles"
                    % format_paise(10_000),
        bank_txn_id=m.bank_txn_id, payment_ids=sorted(m.payment_ids),
        confidence=0.99, expected_rule=Rule.AMOUNT_MISMATCH,
        claimed=claimed - set(m.payment_ids), mutate_txn=doctored))

    # 2. A hallucinated payment id -- the model invents a record.
    m = usable[0]
    attacks.append(Attack(
        name="hallucinated payment id",
        description="a payment id that appears nowhere in settlements.csv",
        bank_txn_id=m.bank_txn_id,
        payment_ids=sorted(m.payment_ids)[:-1] + ["PAY99999"],
        confidence=0.99, expected_rule=Rule.UNKNOWN_PAYMENT,
        claimed=claimed - set(m.payment_ids)))

    # 3. Double-claiming -- the same payment settled against two credits.
    if len(usable) >= 2:
        first, second = usable[0], usable[1]
        stolen = sorted(second.payment_ids)[0]
        attacks.append(Attack(
            name="payment claimed twice",
            description="a payment already settled against %s reused for %s"
                        % (second.bank_txn_id, first.bank_txn_id),
            bank_txn_id=first.bank_txn_id,
            payment_ids=sorted(first.payment_ids)[:-1] + [stolen],
            confidence=0.99, expected_rule=Rule.ALREADY_CLAIMED,
            claimed=(claimed - set(first.payment_ids)) | {stolen}))

    # 4. Date shifted -- a credit attributed to a settlement weeks away.
    m = usable[0]
    txn = ds.bank_by_id[m.bank_txn_id]
    shifted = BankTxn(
        txn_id=txn.txn_id, value_date="2026-06-01", narration=txn.narration,
        credit_paise=txn.credit_paise, debit_paise=0,
        balance_paise=txn.balance_paise)
    attacks.append(Attack(
        name="value date shifted",
        description="credit dated months after the payments settled",
        bank_txn_id=m.bank_txn_id, payment_ids=sorted(m.payment_ids),
        confidence=0.99, expected_rule=Rule.DATE_WINDOW,
        claimed=claimed - set(m.payment_ids), mutate_txn=shifted))

    # 5. A debit row proposed as a settlement.
    debits = [b for b in ds.bank if not b.is_credit]
    if debits:
        m = usable[0]
        attacks.append(Attack(
            name="debit row matched",
            description="payments proposed against a refund debit",
            bank_txn_id=debits[0].txn_id, payment_ids=sorted(m.payment_ids),
            confidence=0.99, expected_rule=Rule.NOT_A_CREDIT,
            claimed=claimed - set(m.payment_ids)))

    # 6. A confident guess below the acceptance threshold.
    m = usable[0]
    attacks.append(Attack(
        name="below confidence threshold",
        description="an otherwise valid match proposed at low confidence",
        bank_txn_id=m.bank_txn_id, payment_ids=sorted(m.payment_ids),
        confidence=0.30, expected_rule=Rule.LOW_CONFIDENCE,
        claimed=claimed - set(m.payment_ids)))

    return attacks


def run_selftest(ds: Dataset, matches, claimed: set[str]) -> tuple[list[dict], bool]:
    """Run every attack. Returns (results, all_caught)."""
    results = []
    all_caught = True

    for attack in build_attacks(ds, matches, claimed):
        # Swap in the doctored row for the duration of this attack only.
        original = ds.bank_by_id.get(attack.bank_txn_id)
        if attack.mutate_txn is not None:
            ds.bank_by_id[attack.bank_txn_id] = attack.mutate_txn
        try:
            verdict = verify(attack.bank_txn_id, attack.payment_ids,
                             attack.confidence, ds, claimed=attack.claimed)
        finally:
            if attack.mutate_txn is not None and original is not None:
                ds.bank_by_id[attack.bank_txn_id] = original

        caught = (not verdict.accepted
                  and verdict.failed_rule is attack.expected_rule)
        all_caught = all_caught and caught
        results.append({
            "attack": attack.name,
            "description": attack.description,
            "expected_rule": str(attack.expected_rule),
            "actual_rule": str(verdict.failed_rule) if verdict.failed_rule else None,
            "caught": caught,
            "accepted": verdict.accepted,
            "reason": verdict.reason,
        })
    return results, all_caught


def format_selftest(results: list[dict]) -> str:
    lines = ["Adversarial self-test of the verification gate",
             "=" * 72,
             "  Each row corrupts a match the pipeline already accepted and",
             "  checks that the gate rejects it for the right reason.",
             ""]
    for r in results:
        status = "CAUGHT" if r["caught"] else "*** MISSED ***"
        lines.append("  %-14s %s" % (status, r["attack"]))
        lines.append("                 %s" % r["description"])
        lines.append("                 rule: %s" % (r["actual_rule"] or "none"))
        if r["reason"]:
            lines.append("                 %s" % r["reason"][:88])
        lines.append("")
    caught = sum(1 for r in results if r["caught"])
    lines.append("  %d/%d corruptions caught" % (caught, len(results)))
    return "\n".join(lines)

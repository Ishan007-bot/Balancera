"""Exception classification.

Everything the pipeline could not resolve gets a named category and a
one-line, plain-English reason a finance analyst could act on. The exception
list is the product -- a finance team's real need is "handle the 90% I don't
want to look at, and hand me a short, correct list of the 10% I do".

Two rules govern this module:

* **Deterministic rules run first and decide most cases.** The evidence that
  classifies an exception -- a narration with no gateway reference, two
  settlements tying on the same amount -- is already structured. Asking a
  model to re-derive it would add cost and a failure mode for nothing.
* **The LLM may categorise, never match.** It is offered only the residue the
  rules could not name, and its answer is constrained to a known category. It
  cannot invent a match here; that path does not exist in this module.

Under ``--no-llm``, or if the call fails, anything unnamed falls back to
``unknown``. That bucket is reported honestly rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ingest import Dataset, Truth
from .models import Case, Exception_
from .money import format_paise

#: Categories an exception can be given. Mirrors the generator's case labels
#: so classification accuracy can be scored against ground truth.
CATEGORIES = tuple(c.value for c in Case) + ("unknown",)


@dataclass
class ClassifiedException:
    exception: Exception_
    category: str
    reason: str
    source: str  # "rule" | "llm" | "fallback"


def classify_deterministic(txn_id: str, ds: Dataset,
                           unresolved_reason: str,
                           ambiguous: list | None = None) -> tuple[str, str] | None:
    """Name an exception from structural evidence alone.

    Returns ``(category, reason)`` or ``None`` if the rules cannot decide,
    in which case the LLM (or the ``unknown`` bucket) takes over.
    """
    txn = ds.bank_by_id.get(txn_id)
    if txn is None:
        return None

    # A debit is never an unmatched settlement -- it is money going out.
    if not txn.is_credit:
        narration = txn.narration.upper()
        if "CHARGEBACK" in narration or "REVERSAL" in narration:
            return (Case.CHARGEBACK_REVERSAL.value,
                    "Debit of %s reverses an earlier settled payment; confirm "
                    "the chargeback against the disputed order."
                    % format_paise(txn.debit_paise))
        if "REFUND" in narration:
            return (Case.REFUND_DEBIT.value,
                    "Refund debit of %s; match it to the refunded order and "
                    "confirm the customer was credited."
                    % format_paise(txn.debit_paise))
        return ("unknown",
                "Debit of %s with no recognised reference; identify the "
                "counterparty." % format_paise(txn.debit_paise))

    # Two or more settlements tie on the same amount and date. This is the
    # case where refusing to choose is the whole point.
    if ambiguous:
        groups = ", ".join(str(a.get("settlement_id", "?")) for a in ambiguous)
        return (Case.AMBIGUOUS_AMOUNT.value,
                "Credit of %s matches %d settlement batches equally well (%s). "
                "Confirm which batch this credit belongs to -- the system "
                "will not guess between them."
                % (format_paise(txn.credit_paise), len(ambiguous), groups))

    # A reference resolved but the amounts disagreed: a real discrepancy.
    if "amounts disagree" in unresolved_reason:
        return (Case.ROUNDING_DRIFT.value,
                "Credit of %s carries a reference that resolves to a "
                "settlement, but the amounts differ. %s Investigate whether "
                "the gateway short-settled."
                % (format_paise(txn.credit_paise),
                   unresolved_reason.split("but ")[-1].capitalize()))

    # No gateway reference at all, and nothing sums to it: most likely not a
    # gateway settlement in the first place.
    from .match_deterministic import extract_reference
    if extract_reference(txn.narration) is None:
        upper = txn.narration.upper()
        if "RAZORPAY" not in upper and "RZPY" not in upper:
            return (Case.FOREIGN_CREDIT.value,
                    "Credit of %s from a non-gateway source (%r); this is "
                    "not a settlement. Identify the payer and post it to the "
                    "correct account."
                    % (format_paise(txn.credit_paise), txn.narration))
        return (Case.MISSING_UTR.value,
                "Gateway credit of %s with no usable reference in the "
                "narration, and no settlement batch sums to it. Request the "
                "settlement report from the gateway."
                % format_paise(txn.credit_paise))

    return None


def classify_unpaid_order(order_id: str, ds: Dataset) -> tuple[str, str]:
    order = ds.orders_by_id.get(order_id)
    amount = format_paise(order.invoice_paise) if order else "?"
    return (Case.UNPAID_ORDER.value,
            "Order %s for %s has no payment record at the gateway. The goods "
            "may have shipped without payment -- confirm before fulfilment."
            % (order_id, amount))


CLASSIFY_SYSTEM_PROMPT = """\
You are a finance operations analyst categorising reconciliation exceptions.

You will be given one unmatched bank transaction. Assign it exactly one \
category from this list and write one sentence an analyst could act on.

Categories:
- foreign_credit: a credit from a non-gateway source; not a settlement at all
- missing_utr: a gateway settlement whose reference is absent from the narration
- mangled_utr: a gateway settlement whose reference is present but corrupted
- ambiguous_amount: matches two or more settlement batches equally well
- rounding_drift: reference resolves but the amount differs slightly
- partial_batch: the credit covers only part of a settlement batch
- weekend_delay: the credit arrived outside the expected settlement window
- refund_debit: a refund going out, not a settlement coming in
- chargeback_reversal: a debit reversing an earlier settled payment
- unknown: none of the above fits

You are categorising only. Do NOT propose which payments this transaction \
matches -- that decision is not yours to make and will be ignored.

Respond with a single JSON object, no prose, no markdown fences:
{"category": "<one of the categories above>", "reason": "<one sentence>"}"""


def build_classify_prompt(txn, unresolved_reason: str) -> str:
    kind = "credit" if txn.is_credit else "debit"
    amount = txn.credit_paise if txn.is_credit else txn.debit_paise
    return (
        "Unmatched bank transaction:\n"
        "  id        : %s\n"
        "  type      : %s\n"
        "  amount    : Rs %s\n"
        "  date      : %s\n"
        "  narration : %s\n"
        "\n"
        "Why the matcher could not resolve it:\n  %s"
        % (txn.txn_id, kind, format_paise(amount), txn.value_date,
           txn.narration, unresolved_reason))


def classify_all(ds: Dataset, unresolved: dict, ambiguous: dict,
                 detected_unpaid: set[str],
                 proposer=None) -> list[ClassifiedException]:
    """Classify every unresolved record. ``proposer`` is optional (offline)."""
    out: list[ClassifiedException] = []

    for txn_id in sorted(unresolved):
        txn = ds.bank_by_id.get(txn_id)
        if txn is None:
            continue
        reason_text = unresolved[txn_id]
        named = classify_deterministic(txn_id, ds, reason_text,
                                       ambiguous.get(txn_id))
        source = "rule"

        if named is None and proposer is not None:
            named = _classify_with_llm(proposer, txn, reason_text)
            source = "llm" if named else "fallback"

        if named is None:
            named = ("unknown",
                     "Credit of %s could not be matched or categorised "
                     "automatically. %s"
                     % (format_paise(txn.credit_paise), reason_text))
            source = "fallback"

        category, reason = named
        out.append(ClassifiedException(
            exception=Exception_(
                record_id=txn_id, record_type="bank_txn", category=category,
                reason=reason,
                amount_paise=txn.credit_paise or txn.debit_paise,
                detail={"narration": txn.narration,
                        "value_date": txn.value_date,
                        "matcher_reason": reason_text}),
            category=category, reason=reason, source=source))

    # Debit rows are never "unresolved" by the matcher because no stage tries
    # to match them -- but they still need to appear on the exception list.
    for txn in ds.debits:
        if txn.txn_id in unresolved:
            continue
        named = classify_deterministic(txn.txn_id, ds, "debit row")
        category, reason = named or ("unknown", "Unidentified debit.")
        out.append(ClassifiedException(
            exception=Exception_(
                record_id=txn.txn_id, record_type="bank_txn",
                category=category, reason=reason,
                amount_paise=txn.debit_paise,
                detail={"narration": txn.narration,
                        "value_date": txn.value_date}),
            category=category, reason=reason, source="rule"))

    for order_id in sorted(detected_unpaid):
        category, reason = classify_unpaid_order(order_id, ds)
        order = ds.orders_by_id.get(order_id)
        out.append(ClassifiedException(
            exception=Exception_(
                record_id=order_id, record_type="order", category=category,
                reason=reason,
                amount_paise=order.invoice_paise if order else 0,
                detail={"customer": order.customer_name if order else "",
                        "created_at": order.created_at if order else ""}),
            category=category, reason=reason, source="rule"))

    return out


def _classify_with_llm(proposer, txn, unresolved_reason: str):
    """Ask the model for a category only. Returns None on any failure."""
    import json

    from .match_llm import strip_fences

    try:
        prompt = build_classify_prompt(txn, unresolved_reason)
        text, _ = proposer._call(CLASSIFY_SYSTEM_PROMPT, prompt, txn.txn_id,
                                 [], 1)
        data = json.loads(strip_fences(text))
        category = data.get("category", "")
        if category not in CATEGORIES:
            return None
        reason = str(data.get("reason", ""))[:400]
        return (category, reason) if reason else None
    except Exception:  # noqa: BLE001 - classification must never break a run
        return None


def classification_accuracy(classified: list[ClassifiedException],
                            truth: Truth) -> dict:
    """Score categories against ground truth -- a bonus metric.

    Only records whose true case is known are scored; the rest are counted
    but excluded from the rate rather than silently assumed correct.
    """
    expected: dict[str, str] = {}
    for tm in truth.matches:
        expected[tm.bank_txn_id] = str(tm.case)
    for txn_id, case in truth.unmatchable_bank.items():
        expected[txn_id] = str(case)
    for txn_id, case in truth.debit_rows.items():
        expected[txn_id] = str(case)
    for order_id, case in truth.unpaid_orders.items():
        expected[order_id] = str(case)

    correct = scored = 0
    mistakes = []
    for c in classified:
        want = expected.get(c.exception.record_id)
        if want is None:
            continue
        scored += 1
        if c.category == want:
            correct += 1
        else:
            mistakes.append({"record_id": c.exception.record_id,
                             "expected": want, "got": c.category})
    return {
        "scored": scored,
        "correct": correct,
        "accuracy": round(correct / scored, 4) if scored else 0.0,
        "mistakes": mistakes,
    }

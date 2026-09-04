"""Core dataclasses and the case taxonomy.

Frozen dataclasses throughout: a reconciliation run must never mutate its
source records. Anything a stage learns is recorded in a Match or Exception,
never written back onto an Order or BankTxn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Case(str, Enum):
    """Generator case labels.

    These are ground truth, and they double as the exception taxonomy in
    Phase 5 so classification accuracy can be scored against them.
    """

    CLEAN_BATCH = "clean_batch"
    WEEKEND_DELAY = "weekend_delay"
    MANGLED_UTR = "mangled_utr"
    MISSING_UTR = "missing_utr"
    AMBIGUOUS_AMOUNT = "ambiguous_amount"
    PARTIAL_BATCH = "partial_batch"
    ROUNDING_DRIFT = "rounding_drift"
    REFUND_DEBIT = "refund_debit"
    CHARGEBACK_REVERSAL = "chargeback_reversal"
    FOREIGN_CREDIT = "foreign_credit"
    UNPAID_ORDER = "unpaid_order"

    def __str__(self) -> str:  # keeps f-strings and CSV writes clean
        return self.value


#: Cases that produce a settlement batch (and therefore a matchable credit).
BATCH_CASES = frozenset({
    Case.CLEAN_BATCH, Case.WEEKEND_DELAY, Case.MANGLED_UTR, Case.MISSING_UTR,
    Case.AMBIGUOUS_AMOUNT, Case.PARTIAL_BATCH, Case.ROUNDING_DRIFT,
})

#: Cases that appear as bank rows matching nothing.
UNMATCHABLE_CASES = frozenset({
    Case.REFUND_DEBIT, Case.CHARGEBACK_REVERSAL, Case.FOREIGN_CREDIT,
})


class OrderStatus(str, Enum):
    PAID = "paid"
    UNPAID = "unpaid"
    REFUNDED = "refunded"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Order:
    order_id: str
    customer_name: str
    invoice_paise: int
    created_at: str  # ISO date, YYYY-MM-DD
    status: OrderStatus


@dataclass(frozen=True, slots=True)
class Payment:
    """A gateway-side payment record, from settlements.csv."""

    payment_id: str
    order_id: str
    gross_paise: int
    fee_paise: int
    gst_paise: int
    net_paise: int
    settlement_id: str
    settled_at: str  # ISO date
    utr: str  # may be "" for missing_utr cases

    def identity_holds(self) -> bool:
        """gross - fee - gst == net. Asserted for every payment by validate."""
        return self.gross_paise - self.fee_paise - self.gst_paise == self.net_paise


@dataclass(frozen=True, slots=True)
class BankTxn:
    txn_id: str
    value_date: str  # ISO date
    narration: str
    credit_paise: int  # 0 on a debit row
    debit_paise: int  # 0 on a credit row
    balance_paise: int

    @property
    def is_credit(self) -> bool:
        return self.credit_paise > 0

    @property
    def signed_amount(self) -> int:
        """Effect on the running balance: positive credit, negative debit."""
        return self.credit_paise - self.debit_paise


@dataclass(frozen=True, slots=True)
class Match:
    """A proposed or accepted link between one bank credit and a payment set.

    ``payment_ids`` is a frozenset because match identity is set equality --
    the order payments were discovered in carries no meaning, and the eval
    harness compares against truth by exact set equality.
    """

    bank_txn_id: str
    payment_ids: frozenset[str]
    stage: str  # which stage produced it: "reference" | "group_sum" | ...
    confidence: float = 1.0
    reasoning: str = ""

    def key(self) -> tuple[str, frozenset[str]]:
        return (self.bank_txn_id, self.payment_ids)


@dataclass(frozen=True, slots=True)
class Exception_:
    """An unresolved record, surfaced for a human.

    Named with a trailing underscore to avoid shadowing the builtin. The
    exception list is a deliverable, not a failure -- every field here exists
    to make one line of the final report actionable.
    """

    record_id: str  # bank txn id or order id
    record_type: str  # "bank_txn" | "order"
    category: str  # a Case value, or "unknown"
    reason: str  # plain English, one line, analyst-actionable
    amount_paise: int = 0
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TruthMatch:
    """One expected match from truth.json."""

    bank_txn_id: str
    payment_ids: frozenset[str]
    case: Case

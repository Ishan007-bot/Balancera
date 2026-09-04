"""CSV -> dataclasses, with normalisation.

Everything downstream reads the objects this module produces, never the CSVs
directly. Amounts are parsed to ``int`` paise here and stay integers for the
rest of the run; a float entering at this boundary would silently break every
later equality comparison.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .models import BankTxn, Case, Order, OrderStatus, Payment, TruthMatch


@dataclass
class Dataset:
    """Everything one reconciliation run operates on."""

    orders: list[Order]
    payments: list[Payment]
    bank: list[BankTxn]

    # Indexes, built once. Stages hit these constantly; rebuilding them per
    # stage would dominate the runtime we report as throughput.
    orders_by_id: dict[str, Order] = field(default_factory=dict)
    payments_by_id: dict[str, Payment] = field(default_factory=dict)
    bank_by_id: dict[str, BankTxn] = field(default_factory=dict)
    payments_by_settlement: dict[str, list[Payment]] = field(default_factory=dict)
    payments_by_utr: dict[str, list[Payment]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.orders_by_id = {o.order_id: o for o in self.orders}
        self.payments_by_id = {p.payment_id: p for p in self.payments}
        self.bank_by_id = {b.txn_id: b for b in self.bank}
        for p in self.payments:
            self.payments_by_settlement.setdefault(p.settlement_id, []).append(p)
            if p.utr:
                # Keyed by digits only: narration templates disagree about the
                # "UTR" prefix, so comparing raw strings would miss references
                # that are in fact identical.
                key = "".join(ch for ch in p.utr if ch.isdigit())
                self.payments_by_utr.setdefault(key, []).append(p)

    @property
    def credits(self) -> list[BankTxn]:
        """Credit rows only -- the matchable side of the bank statement."""
        return [b for b in self.bank if b.is_credit]

    @property
    def debits(self) -> list[BankTxn]:
        return [b for b in self.bank if not b.is_credit]

    def settlement_of(self, payment_id: str) -> str | None:
        p = self.payments_by_id.get(payment_id)
        return p.settlement_id if p else None

    def net_sum(self, payment_ids) -> int:
        """Sum net_paise over payment ids, recomputed from source.

        The verification gate uses this rather than trusting any arithmetic
        that arrives from outside -- notably the LLM's.
        """
        return sum(self.payments_by_id[pid].net_paise for pid in payment_ids)


@dataclass
class Truth:
    """Parsed truth.json. Used only by evaluate.py, never by the matchers."""

    seed: int
    hard_ratio: float
    matches: list[TruthMatch]
    unmatchable_bank: dict[str, Case]  # txn_id -> case
    debit_rows: dict[str, Case]
    unpaid_orders: dict[str, Case]  # order_id -> case
    opening_balance_paise: int = 0
    counts: dict = field(default_factory=dict)

    @property
    def by_txn(self) -> dict[str, TruthMatch]:
        return {m.bank_txn_id: m for m in self.matches}

    @property
    def matchable_txn_ids(self) -> set[str]:
        return {m.bank_txn_id for m in self.matches}


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _int(value: str, field_name: str, row_id: str) -> int:
    """Parse an integer paise field, refusing anything fractional.

    A float here would be a data-quality problem worth failing loudly on, not
    something to coerce quietly.
    """
    text = (value or "").strip()
    if not text:
        return 0
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError("%s: %s=%r is not integer paise" %
                         (row_id, field_name, value)) from exc


def load_dataset(data_dir) -> Dataset:
    d = Path(data_dir)
    orders = [
        Order(
            order_id=r["order_id"].strip(),
            customer_name=r["customer_name"].strip(),
            invoice_paise=_int(r["invoice_paise"], "invoice_paise", r["order_id"]),
            created_at=r["created_at"].strip(),
            status=OrderStatus(r["status"].strip()),
        )
        for r in _rows(d / "orders.csv")
    ]
    payments = [
        Payment(
            payment_id=r["payment_id"].strip(),
            order_id=r["order_id"].strip(),
            gross_paise=_int(r["gross_paise"], "gross_paise", r["payment_id"]),
            fee_paise=_int(r["fee_paise"], "fee_paise", r["payment_id"]),
            gst_paise=_int(r["gst_paise"], "gst_paise", r["payment_id"]),
            net_paise=_int(r["net_paise"], "net_paise", r["payment_id"]),
            settlement_id=r["settlement_id"].strip(),
            settled_at=r["settled_at"].strip(),
            # Normalise case and spacing; a UTR is a reference, not free text.
            utr=r["utr"].strip().upper(),
        )
        for r in _rows(d / "settlements.csv")
    ]
    bank = [
        BankTxn(
            txn_id=r["txn_id"].strip(),
            value_date=r["value_date"].strip(),
            # Collapse runs of whitespace but keep the text otherwise intact --
            # the messiness is the point, and over-cleaning would make the
            # reference-extraction problem artificially easy.
            narration=" ".join(r["narration"].split()),
            credit_paise=_int(r["credit_paise"], "credit_paise", r["txn_id"]),
            debit_paise=_int(r["debit_paise"], "debit_paise", r["txn_id"]),
            balance_paise=_int(r["balance_paise"], "balance_paise", r["txn_id"]),
        )
        for r in _rows(d / "bank.csv")
    ]
    return Dataset(orders=orders, payments=payments, bank=bank)


def load_truth(data_dir) -> Truth:
    raw = json.loads((Path(data_dir) / "truth.json").read_text(encoding="utf-8"))
    return Truth(
        seed=raw["seed"],
        hard_ratio=raw["hard_ratio"],
        matches=[
            TruthMatch(
                bank_txn_id=m["bank_txn_id"],
                payment_ids=frozenset(m["payment_ids"]),
                case=Case(m["case"]),
            )
            for m in raw["matches"]
        ],
        unmatchable_bank={u["bank_txn_id"]: Case(u["case"])
                          for u in raw.get("unmatchable_bank", [])},
        debit_rows={u["bank_txn_id"]: Case(u["case"])
                    for u in raw.get("debit_rows", [])},
        unpaid_orders={u["order_id"]: Case(u["case"])
                       for u in raw.get("unpaid_orders", [])},
        opening_balance_paise=raw.get("opening_balance_paise", 0),
        counts=raw.get("counts", {}),
    )


def parse_date(iso: str) -> date:
    return date.fromisoformat(iso)

"""Verification gate tests.

SPEC section 14 priority 1 -- the highest-value tests in the repository. Each
rejection rule is tested in isolation, plus the properties that matter more
than any individual rule: the gate cannot be talked into accepting bad
arithmetic by confidence, and it never trusts a stated sum.
"""

import pytest

from recon.ingest import Dataset
from recon.models import BankTxn, Order, OrderStatus, Payment
from recon.verify import (
    CROSS_SETTLEMENT_CONFIDENCE, GateResult, Rule, verify, verify_all,
)


def _payment(pid, net, stl="STL0001", settled="2026-01-05"):
    """A payment whose gross/fee/gst are consistent with its net."""
    return Payment(payment_id=pid, order_id="ORD_" + pid, gross_paise=net,
                   fee_paise=0, gst_paise=0, net_paise=net,
                   settlement_id=stl, settled_at=settled, utr="UTR1234567")


def _credit(txn_id="BNK1", amount=1000, value_date="2026-01-07"):
    return BankTxn(txn_id=txn_id, value_date=value_date, narration="TEST",
                   credit_paise=amount, debit_paise=0, balance_paise=0)


@pytest.fixture
def ds():
    """300 + 700 = 1000, matching BNK1. PAY3 belongs to another settlement."""
    payments = [
        _payment("PAY1", 300), _payment("PAY2", 700),
        _payment("PAY3", 500, stl="STL0002"),
        _payment("PAY4", 400, settled="2025-12-01"),  # settled long before
    ]
    bank = [
        _credit("BNK1", 1000),
        _credit("BNK2", 500),
        BankTxn("BNK3", "2026-01-07", "REFUND", 0, 250, 0),  # a debit row
    ]
    return Dataset(orders=[Order("ORD_PAY1", "A", 300, "2026-01-01",
                                 OrderStatus.PAID)],
                   payments=payments, bank=bank)


class TestAcceptance:
    def test_a_correct_proposal_is_accepted(self, ds):
        v = verify("BNK1", ["PAY1", "PAY2"], 0.9, ds, claimed=set())
        assert v.accepted
        assert v.failed_rule is None
        # 1000 paise is Rs 10.00 -- amounts are shown in rupees, not paise.
        assert "10.00" in v.reason

    def test_rounding_drift_within_tolerance_is_accepted(self, ds):
        """1 paise of fee-rounding drift must not block a real match."""
        ds.bank_by_id["BNK1"] = _credit("BNK1", 1001)
        v = verify("BNK1", ["PAY1", "PAY2"], 0.9, ds, claimed=set())
        assert v.accepted

    def test_accepted_verdict_records_the_arithmetic(self, ds):
        v = verify("BNK1", ["PAY1", "PAY2"], 0.9, ds, claimed=set())
        assert v.detail["proposed_sum_paise"] == 1000
        assert v.detail["credit_paise"] == 1000
        assert v.detail["delta_paise"] == 0


class TestEachRejectionRule:
    """One test per rule, in isolation."""

    def test_empty_proposal(self, ds):
        v = verify("BNK1", [], 0.9, ds, claimed=set())
        assert not v.accepted and v.failed_rule is Rule.EMPTY_PROPOSAL

    def test_unknown_payment_id(self, ds):
        """The rule that catches a hallucinated identifier."""
        v = verify("BNK1", ["PAY1", "PAY99999"], 0.95, ds, claimed=set())
        assert not v.accepted and v.failed_rule is Rule.UNKNOWN_PAYMENT
        assert "PAY99999" in v.reason

    def test_already_claimed_payment(self, ds):
        v = verify("BNK1", ["PAY1", "PAY2"], 0.9, ds, claimed={"PAY2"})
        assert not v.accepted and v.failed_rule is Rule.ALREADY_CLAIMED
        assert "PAY2" in v.reason

    def test_amount_mismatch(self, ds):
        v = verify("BNK1", ["PAY1"], 0.99, ds, claimed=set())
        assert not v.accepted and v.failed_rule is Rule.AMOUNT_MISMATCH
        assert v.detail["proposed_sum_paise"] == 300
        assert v.detail["credit_paise"] == 1000
        assert v.detail["delta_paise"] == -700

    def test_date_window(self, ds):
        """PAY4 settled in December against a January credit."""
        ds.bank_by_id["BNK2"] = _credit("BNK2", 400)
        v = verify("BNK2", ["PAY4"], 0.9, ds, claimed=set())
        assert not v.accepted and v.failed_rule is Rule.DATE_WINDOW
        assert v.detail["lag_days"] > 3

    def test_credit_dated_before_settlement_is_rejected(self, ds):
        """Money cannot reach the bank before the gateway releases it."""
        ds.bank_by_id["BNK2"] = _credit("BNK2", 300, value_date="2026-01-01")
        v = verify("BNK2", ["PAY1"], 0.9, ds, claimed=set())
        assert not v.accepted and v.failed_rule is Rule.DATE_WINDOW

    def test_cross_settlement_below_raised_threshold(self, ds):
        """Spanning settlements is allowed but needs stronger evidence."""
        ds.bank_by_id["BNK2"] = _credit("BNK2", 800)
        v = verify("BNK2", ["PAY1", "PAY3"], 0.75, ds, claimed=set())
        assert not v.accepted and v.failed_rule is Rule.CROSS_SETTLEMENT
        assert v.detail["settlement_ids"] == ["STL0001", "STL0002"]

    def test_cross_settlement_above_raised_threshold_is_accepted(self, ds):
        ds.bank_by_id["BNK2"] = _credit("BNK2", 800)
        v = verify("BNK2", ["PAY1", "PAY3"], CROSS_SETTLEMENT_CONFIDENCE,
                   ds, claimed=set())
        assert v.accepted

    def test_low_confidence(self, ds):
        v = verify("BNK1", ["PAY1", "PAY2"], 0.5, ds, claimed=set())
        assert not v.accepted and v.failed_rule is Rule.LOW_CONFIDENCE

    def test_conflict_with_an_already_matched_credit(self, ds):
        v = verify("BNK1", ["PAY1", "PAY2"], 0.9, ds, claimed=set(),
                   accepted_txns={"BNK1"})
        assert not v.accepted and v.failed_rule is Rule.CONFLICT

    def test_debit_row_cannot_be_matched(self, ds):
        v = verify("BNK3", ["PAY1"], 0.99, ds, claimed=set())
        assert not v.accepted and v.failed_rule is Rule.NOT_A_CREDIT

    def test_nonexistent_transaction(self, ds):
        v = verify("BNK9999", ["PAY1"], 0.99, ds, claimed=set())
        assert not v.accepted and v.failed_rule is Rule.NOT_A_CREDIT


class TestTheGateCannotBeTalkedInto:
    """Confidence is not evidence. These are the assertions that make the
    architecture defensible rather than decorative."""

    def test_maximum_confidence_does_not_excuse_bad_arithmetic(self, ds):
        v = verify("BNK1", ["PAY1"], 1.0, ds, claimed=set())
        assert not v.accepted and v.failed_rule is Rule.AMOUNT_MISMATCH

    def test_maximum_confidence_does_not_excuse_a_hallucinated_id(self, ds):
        v = verify("BNK1", ["PAY_INVENTED"], 1.0, ds, claimed=set())
        assert not v.accepted and v.failed_rule is Rule.UNKNOWN_PAYMENT

    def test_maximum_confidence_does_not_excuse_a_claimed_payment(self, ds):
        v = verify("BNK1", ["PAY1", "PAY2"], 1.0, ds, claimed={"PAY1"})
        assert not v.accepted and v.failed_rule is Rule.ALREADY_CLAIMED

    def test_the_stated_sum_is_never_trusted(self, ds):
        """A proposal carrying its own (wrong) arithmetic must still fail.

        The gate recomputes from payments_by_id; nothing the proposer asserts
        about totals is read at all.
        """
        class LyingProposal:
            bank_txn_id = "BNK1"
            proposed_payment_ids = ["PAY1"]  # 300 paise against a 1000 credit
            confidence = 0.99
            proposed_sum_paise = 1000  # the lie
            reasoning = "these sum to exactly 1000"

        result = verify_all([LyingProposal()], ds, claimed=set())
        assert not result.accepted
        assert result.rejected[0].failed_rule is Rule.AMOUNT_MISMATCH
        assert result.rejected[0].detail["proposed_sum_paise"] == 300


class TestRejectionLog:
    """The log is a deliverable, so its content is tested, not just its shape."""

    def test_record_contains_amounts_not_just_a_rule_name(self, ds):
        v = verify("BNK1", ["PAY1"], 0.99, ds, claimed=set())
        record = v.as_log_record()
        assert record["failed_rule"] == "amount_mismatch"
        assert record["proposed_sum_paise"] == 300
        assert record["credit_paise"] == 1000
        assert "3.00" in record["reason"] and "10.00" in record["reason"]

    def test_record_lists_the_proposed_payments(self, ds):
        v = verify("BNK1", ["PAY2", "PAY1"], 0.5, ds, claimed=set())
        assert v.as_log_record()["proposed_payment_ids"] == ["PAY1", "PAY2"]

    def test_reason_is_readable_aloud(self, ds):
        """The point of the log is that it can be read out on camera."""
        v = verify("BNK1", ["PAY1"], 0.99, ds, claimed=set())
        assert "sum to" in v.reason and "credit is" in v.reason


class TestVerifyAll:
    def test_accepted_matches_consume_their_payments(self, ds):
        """A payment accepted for one credit is unavailable to the next."""
        class P:
            def __init__(self, txn, ids, conf=0.9):
                self.bank_txn_id, self.proposed_payment_ids = txn, ids
                self.confidence = conf

        ds.bank_by_id["BNK2"] = _credit("BNK2", 300)
        result = verify_all([P("BNK1", ["PAY1", "PAY2"]), P("BNK2", ["PAY1"])],
                            ds, claimed=set())
        assert len(result.accepted) == 1
        assert result.rejected[0].failed_rule is Rule.ALREADY_CLAIMED

    def test_summary_counts_rejections_by_rule(self, ds):
        class P:
            def __init__(self, txn, ids, conf):
                self.bank_txn_id, self.proposed_payment_ids = txn, ids
                self.confidence = conf

        result = verify_all([P("BNK1", ["PAY1"], 0.9),        # amount
                             P("BNK1", ["PAY1", "PAY2"], 0.1),  # confidence
                             P("BNK1", ["NOPE"], 0.9)],         # unknown
                            ds, claimed=set())
        summary = result.summary()
        assert summary["accepted"] == 0
        assert summary["rejected"] == 3
        assert summary["rejections_by_rule"] == {
            "amount_mismatch": 1, "low_confidence": 1, "unknown_payment": 1}

    def test_empty_input(self, ds):
        assert verify_all([], ds, claimed=set()).summary()["proposals_verified"] == 0


def test_verify_module_has_no_llm_awareness():
    """SPEC: the gate must be independently testable and must not import
    match_llm. A checker that shares assumptions with the thing it checks is
    not checking anything."""
    import ast
    import pathlib

    source = pathlib.Path("recon/verify.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert not any("match_llm" in m or "anthropic" in m for m in imported), \
        "verify.py imports the proposer it is supposed to check: %s" % imported
    # The module may *mention* match_llm in prose to explain why it does not
    # import it; what matters is that no import statement exists.
    import_lines = [line for line in source.split("\n")
                    if line.strip().startswith(("import ", "from "))]
    joined = "\n".join(import_lines)
    assert "match_llm" not in joined
    assert "anthropic" not in joined

"""Eval harness tests.

The metric definitions are what the submission's credibility rests on, so
they are tested directly against hand-built cases rather than only through
the pipeline. In particular: a wrong match must never improve a score.
"""

import pytest

from recon.evaluate import evaluate, per_case_breakdown
from recon.ingest import Dataset, Truth
from recon.models import BankTxn, Case, Match, Order, OrderStatus, Payment, TruthMatch


def _payment(pid, net, stl="STL0001"):
    return Payment(payment_id=pid, order_id="ORD_" + pid, gross_paise=net,
                   fee_paise=0, gst_paise=0, net_paise=net,
                   settlement_id=stl, settled_at="2026-01-05", utr="UTR1234567")


def _credit(txn_id, amount):
    return BankTxn(txn_id=txn_id, value_date="2026-01-07", narration="X",
                   credit_paise=amount, debit_paise=0, balance_paise=0)


@pytest.fixture
def fixture():
    """Two truth matches, one foreign credit, one unpaid order."""
    payments = [_payment("PAY1", 100), _payment("PAY2", 200),
                _payment("PAY3", 300, "STL0002")]
    orders = [Order("ORD_PAY1", "A", 100, "2026-01-01", OrderStatus.PAID),
              Order("ORD_UNPAID", "B", 500, "2026-01-01", OrderStatus.UNPAID)]
    bank = [_credit("BNK1", 300), _credit("BNK2", 300), _credit("BNK3", 999)]
    ds = Dataset(orders=orders, payments=payments, bank=bank)
    truth = Truth(
        seed=42, hard_ratio=0.4,
        matches=[
            TruthMatch("BNK1", frozenset({"PAY1", "PAY2"}), Case.CLEAN_BATCH),
            TruthMatch("BNK2", frozenset({"PAY3"}), Case.PARTIAL_BATCH),
        ],
        unmatchable_bank={"BNK3": Case.FOREIGN_CREDIT},
        debit_rows={},
        unpaid_orders={"ORD_UNPAID": Case.UNPAID_ORDER},
    )
    return ds, truth


class TestCoreDefinitions:
    def test_exact_set_equality_is_a_true_positive(self, fixture):
        ds, truth = fixture
        m = [Match("BNK1", frozenset({"PAY1", "PAY2"}), "group_sum")]
        result = evaluate(m, truth, ds)
        assert result.true_positives == 1
        assert result.false_positives == 0

    def test_overlapping_set_is_a_partial_and_a_false_positive(self, fixture):
        """Partial matches count against strict precision but are reported
        separately -- hiding them would be dishonest."""
        ds, truth = fixture
        m = [Match("BNK1", frozenset({"PAY1"}), "group_sum")]
        result = evaluate(m, truth, ds)
        assert result.true_positives == 0
        assert result.false_positives == 1
        assert result.partial_matches == 1

    def test_disjoint_set_is_a_false_positive_but_not_partial(self, fixture):
        ds, truth = fixture
        m = [Match("BNK1", frozenset({"PAY3"}), "group_sum")]
        result = evaluate(m, truth, ds)
        assert result.false_positives == 1
        assert result.partial_matches == 0

    def test_match_against_foreign_credit_is_a_forced_match_error(self, fixture):
        """The most important number on the page. It must be zero."""
        ds, truth = fixture
        m = [Match("BNK3", frozenset({"PAY1"}), "group_sum")]
        result = evaluate(m, truth, ds)
        assert result.forced_match_errors == 1
        assert result.false_positives == 1  # forced errors are a subset of FP

    def test_perfect_run(self, fixture):
        ds, truth = fixture
        m = [Match("BNK1", frozenset({"PAY1", "PAY2"}), "group_sum"),
             Match("BNK2", frozenset({"PAY3"}), "subset_sum")]
        result = evaluate(m, truth, ds, detected_unpaid={"ORD_UNPAID"})
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.auto_match_rate == 1.0
        assert result.forced_match_errors == 0
        assert result.unpaid_order_recall == 1.0
        assert result.foreign_credits_left_unmatched == 1


class TestGamingResistance:
    """A matcher that guesses must not be able to score well."""

    def test_forcing_a_match_on_everything_scores_badly(self, fixture):
        """The correction agreed during planning: auto-match rate counts only
        CORRECT matches, so guessing on every row cannot reach 100%."""
        ds, truth = fixture
        m = [Match("BNK1", frozenset({"PAY1"}), "guess"),
             Match("BNK2", frozenset({"PAY2"}), "guess"),
             Match("BNK3", frozenset({"PAY3"}), "guess")]
        result = evaluate(m, truth, ds)
        assert result.auto_match_rate == 0.0
        assert result.precision == 0.0
        assert result.forced_match_errors == 1

    def test_abstaining_entirely_scores_zero_recall_but_perfect_safety(self, fixture):
        ds, truth = fixture
        result = evaluate([], truth, ds)
        assert result.recall == 0.0
        assert result.forced_match_errors == 0
        assert result.foreign_credits_left_unmatched == 1

    def test_precision_is_zero_not_undefined_with_no_proposals(self, fixture):
        ds, truth = fixture
        assert evaluate([], truth, ds).precision == 0.0


class TestReporting:
    def test_missed_transactions_are_listed(self, fixture):
        ds, truth = fixture
        m = [Match("BNK1", frozenset({"PAY1", "PAY2"}), "group_sum")]
        result = evaluate(m, truth, ds)
        assert result.detail["missed_txns"] == ["BNK2"]

    def test_partial_detail_names_missing_and_extra(self, fixture):
        ds, truth = fixture
        m = [Match("BNK1", frozenset({"PAY1", "PAY3"}), "group_sum")]
        result = evaluate(m, truth, ds)
        detail = result.detail["partial_matches"][0]
        assert detail["missing"] == ["PAY2"]
        assert detail["extra"] == ["PAY3"]

    def test_per_case_breakdown(self, fixture):
        ds, truth = fixture
        m = [Match("BNK1", frozenset({"PAY1", "PAY2"}), "group_sum")]
        rows = per_case_breakdown(m, truth)
        assert rows["clean_batch"] == {"total": 1, "correct": 1, "wrong": 0,
                                       "rate": 1.0}
        assert rows["partial_batch"]["correct"] == 0

    def test_unpaid_order_detection_is_scored(self, fixture):
        ds, truth = fixture
        assert evaluate([], truth, ds, detected_unpaid=set()).unpaid_order_recall == 0.0
        assert evaluate([], truth, ds,
                        detected_unpaid={"ORD_UNPAID"}).unpaid_order_recall == 1.0

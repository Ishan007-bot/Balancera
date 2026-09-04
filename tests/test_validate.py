"""Validator tests.

SPEC section 14 priority 5: deliberately corrupt a truth file and assert it
is caught. A validator that only ever passes proves nothing -- these tests
are what make its green result meaningful.

Each test corrupts exactly one thing and asserts the specific rule fires.
"""

import csv
import json

import pytest

from recon.generate import generate
from recon.validate import extract_reference, validate


@pytest.fixture
def data(tmp_path):
    """A fresh, valid dataset per test -- each one mutates it."""
    generate(42, 0.4, tmp_path)
    return tmp_path


def _read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys(), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _truth(data):
    return json.loads((data / "truth.json").read_text(encoding="utf-8"))


def _save_truth(data, truth):
    (data / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")


def _fails_with(data, marker):
    """Assert validate reports at least one failure containing ``marker``."""
    failures = validate(data)
    assert failures, "validator passed corrupted data (expected %s)" % marker
    assert any(marker in f for f in failures), \
        "expected %r, got: %s" % (marker, failures)
    return failures


def test_clean_data_passes(data):
    assert validate(data) == []


def test_missing_file_is_caught(data):
    (data / "bank.csv").unlink()
    _fails_with(data, "MISSING FILE")


def test_broken_payment_identity(data):
    rows = _read(data / "settlements.csv")
    rows[0]["net_paise"] = str(int(rows[0]["net_paise"]) + 1)
    _write(data / "settlements.csv", rows)
    _fails_with(data, "PAYMENT IDENTITY")


def test_credit_sum_mismatch(data):
    rows = _read(data / "bank.csv")
    for r in rows:
        if int(r["credit_paise"]) > 0:
            r["credit_paise"] = str(int(r["credit_paise"]) + 500)
            break
    _write(data / "bank.csv", rows)
    _fails_with(data, "CREDIT SUM")


def test_balance_discontinuity(data):
    rows = _read(data / "bank.csv")
    rows[3]["balance_paise"] = str(int(rows[3]["balance_paise"]) + 100)
    _write(data / "bank.csv", rows)
    _fails_with(data, "BALANCE")


def test_payment_claimed_by_two_matches(data):
    truth = _truth(data)
    truth["matches"][1]["payment_ids"].append(truth["matches"][0]["payment_ids"][0])
    _save_truth(data, truth)
    _fails_with(data, "DOUBLE CLAIM")


def test_truth_references_unknown_payment(data):
    truth = _truth(data)
    truth["matches"][0]["payment_ids"].append("PAY99999")
    _save_truth(data, truth)
    _fails_with(data, "TRUTH REF")


def test_truth_references_unknown_bank_txn(data):
    truth = _truth(data)
    truth["matches"][0]["bank_txn_id"] = "BNK9999"
    _save_truth(data, truth)
    _fails_with(data, "TRUTH REF")


def test_foreign_credit_given_a_matchable_reference(data):
    """The single most important check: if a foreign credit could be matched
    by reference, stage 1 would confidently pair a transaction that
    corresponds to nothing."""
    truth = _truth(data)
    target = truth["unmatchable_bank"][0]["bank_txn_id"]
    rows = _read(data / "bank.csv")
    for r in rows:
        if r["txn_id"] == target:
            r["narration"] = "NEFT-RAZORPAYSOFT-UTR8842910-SETTLEMENT"
    _write(data / "bank.csv", rows)
    _fails_with(data, "FOREIGN CREDIT REGEX")


def test_case_counts_disagree(data):
    truth = _truth(data)
    truth["case_counts"]["clean_batch"] += 5
    _save_truth(data, truth)
    _fails_with(data, "CASE COUNT")


def test_partial_batch_withheld_net_not_unique(data):
    """Two payments with the same net means two subsets tie, so the matcher
    reports ambiguity and the truth becomes unmatchable."""
    truth = _truth(data)
    pm = next(m for m in truth["matches"] if m["case"] == "partial_batch")
    rows = _read(data / "settlements.csv")
    withheld = next(r for r in rows if r["payment_id"] == pm["withheld_payment_id"])
    twin = next(r for r in rows if r["settlement_id"] == pm["settlement_id"]
                and r["payment_id"] != withheld["payment_id"])
    for field in ("gross_paise", "fee_paise", "gst_paise", "net_paise"):
        twin[field] = withheld[field]
    _write(data / "settlements.csv", rows)
    failures = validate(data)
    assert failures, "validator passed an unmatchable partial_batch"


def test_unpaid_order_that_actually_has_a_payment(data):
    truth = _truth(data)
    paid = _read(data / "settlements.csv")[0]["order_id"]
    truth["unpaid_orders"].append({"order_id": paid, "case": "unpaid_order"})
    _save_truth(data, truth)
    _fails_with(data, "UNPAID ORDER")


def test_orphan_bank_row(data):
    truth = _truth(data)
    truth["unmatchable_bank"] = []
    _save_truth(data, truth)
    _fails_with(data, "ORPHAN BANK ROW")


def test_inert_weekend_delay_is_caught(data):
    """Regression for a bug that shipped: value_date == settled_at made the
    weekend_delay case match trivially, silently inflating the baseline."""
    truth = _truth(data)
    pays = {p["payment_id"]: p for p in _read(data / "settlements.csv")}
    targets = {m["bank_txn_id"]: pays[m["payment_ids"][0]]["settled_at"]
               for m in truth["matches"] if m["case"] == "weekend_delay"}
    rows = _read(data / "bank.csv")
    for r in rows:
        if r["txn_id"] in targets:
            r["value_date"] = targets[r["txn_id"]]
    _write(data / "bank.csv", rows)
    _fails_with(data, "WEEKEND DELAY INERT")


def test_value_date_before_settled_at(data):
    truth = _truth(data)
    txn = truth["matches"][0]["bank_txn_id"]
    rows = _read(data / "bank.csv")
    for r in rows:
        if r["txn_id"] == txn:
            r["value_date"] = "2020-01-01"
    _write(data / "bank.csv", rows)
    _fails_with(data, "DATE ORDER")


def test_clean_batch_may_not_carry_drift(data):
    truth = _truth(data)
    m = next(m for m in truth["matches"] if m["case"] == "clean_batch")
    m["drift_paise"] = 3
    _save_truth(data, truth)
    _fails_with(data, "CLEAN BATCH DRIFT")


class TestReferenceExtraction:
    """The regex is shared with the Phase 2 matcher, so its behaviour is
    part of the ground-truth contract, not a matcher implementation detail."""

    def test_extracts_from_each_template(self):
        assert extract_reference("NEFT-RAZORPAYSOFT-UTR8842910-SETTLEMENT")
        assert extract_reference("IMPS/UTR3592974/RAZORPAY/COLLECTION")
        assert extract_reference("RTGS RZPYSOFT UTR6120253 NET STLMT")
        assert extract_reference("UPI-RAZORPAY-UTR8795419")
        assert extract_reference("UTR8973915 RAZORPAYSOFTWA")

    def test_rejects_foreign_narrations(self):
        assert extract_reference("NEFT-ACMESUPPLIES-INV4471") is None
        assert extract_reference("NEFT-VENDORREFUND-BILL2290") is None

    def test_rejects_no_reference_narration(self):
        assert extract_reference("NEFT-RAZORPAYSOFT-SETTLEMENT-BULK") is None

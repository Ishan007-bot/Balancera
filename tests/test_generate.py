"""Generator tests.

SPEC section 14 priority 4: same seed, identical output. Plus the structural
guarantees each hard case must provide -- a case that does not actually
exercise what it claims to is worse than no case at all, because it inflates
the baseline while looking like rigour.
"""

import csv
import json
from datetime import date

import pytest

from recon.generate import build_plan, generate
from recon.models import Case


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """Generate one dataset at defaults, shared across this module."""
    out = tmp_path_factory.mktemp("data")
    generate(42, 0.4, out)
    return out


def _read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class TestDeterminism:
    """--seed 42 must reproduce byte-identical files. This is graded."""

    def test_same_seed_identical_bytes(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        generate(42, 0.4, a)
        generate(42, 0.4, b)
        for name in ["orders.csv", "settlements.csv", "bank.csv", "truth.json"]:
            assert (a / name).read_bytes() == (b / name).read_bytes(), name

    def test_different_seed_differs(self, tmp_path):
        """Proves the seed is actually wired through, not that output is constant."""
        a, b = tmp_path / "a", tmp_path / "b"
        generate(42, 0.4, a)
        generate(99, 0.4, b)
        assert (a / "bank.csv").read_bytes() != (b / "bank.csv").read_bytes()

    def test_plan_is_deterministic(self):
        p1 = build_plan(42, 0.4)
        p2 = build_plan(42, 0.4)
        assert [b.settlement_id for b in p1.batches] == \
               [b.settlement_id for b in p2.batches]
        assert [b.case for b in p1.batches] == [b.case for b in p2.batches]
        assert [b.credit_paise() for b in p1.batches] == \
               [b.credit_paise() for b in p2.batches]

    def test_files_use_lf_newlines(self, data):
        """CRLF would make the determinism diff platform-dependent."""
        raw = (data / "bank.csv").read_bytes()
        assert b"\r\n" not in raw


class TestCaseMix:
    def test_sixty_percent_clean_at_default_ratio(self, data):
        truth = json.loads((data / "truth.json").read_text())
        counts = truth["case_counts"]
        total = sum(counts.values())
        assert total == 30
        assert counts["clean_batch"] == 18
        assert counts["clean_batch"] / total == pytest.approx(0.6)

    def test_every_spec_case_is_present(self, data):
        truth = json.loads((data / "truth.json").read_text())
        counts = truth["case_counts"]
        for case in ["weekend_delay", "mangled_utr", "missing_utr",
                     "ambiguous_amount", "partial_batch", "rounding_drift"]:
            assert counts.get(case) == 2, case
        assert len(truth["unmatchable_bank"]) == 2  # foreign_credit
        assert len(truth["unpaid_orders"]) == 3
        debits = [d["case"] for d in truth["debit_rows"]]
        assert debits.count("refund_debit") == 2
        assert debits.count("chargeback_reversal") == 1

    def test_hard_ratio_scales(self, tmp_path):
        shares = {}
        for ratio in (0.2, 0.4, 0.6):
            out = tmp_path / str(ratio)
            generate(42, ratio, out)
            counts = json.loads((out / "truth.json").read_text())["case_counts"]
            total = sum(counts.values())
            shares[ratio] = counts.get("clean_batch", 0) / total
        # More hardness must mean strictly fewer clean batches.
        assert shares[0.2] > shares[0.4] > shares[0.6]

    def test_ambiguous_amount_only_appears_in_pairs(self, tmp_path):
        """A single ambiguous batch cannot create a tie -- it must be 0 or even."""
        for ratio in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
            out = tmp_path / ("r%s" % ratio)
            generate(42, ratio, out)
            counts = json.loads((out / "truth.json").read_text())["case_counts"]
            assert counts.get("ambiguous_amount", 0) % 2 == 0, ratio


class TestHardCasesActuallyBite:
    """Each hard case must exercise the thing it claims to exercise."""

    def test_weekend_delay_exceeds_naive_two_day_window(self, data):
        """Regression: this shipped inert once. value_date - settled_at was 0
        for every batch, so the case matched trivially and proved nothing."""
        truth = json.loads((data / "truth.json").read_text())
        pays = {p["payment_id"]: p for p in _read(data / "settlements.csv")}
        bank = {b["txn_id"]: b for b in _read(data / "bank.csv")}
        lags = {}
        for m in truth["matches"]:
            settled = date.fromisoformat(pays[m["payment_ids"][0]]["settled_at"])
            value = date.fromisoformat(bank[m["bank_txn_id"]]["value_date"])
            lags.setdefault(m["case"], set()).add((value - settled).days)
        assert lags["weekend_delay"] == {3}, "weekend_delay must exceed T+2"
        for case, seen in lags.items():
            if case != "weekend_delay":
                assert seen == {2}, "%s should settle at T+2, got %s" % (case, seen)

    def test_mangled_utr_is_not_findable_verbatim(self, data):
        truth = json.loads((data / "truth.json").read_text())
        pays = {p["payment_id"]: p for p in _read(data / "settlements.csv")}
        bank = {b["txn_id"]: b for b in _read(data / "bank.csv")}
        for m in truth["matches"]:
            if m["case"] != "mangled_utr":
                continue
            true_utr = pays[m["payment_ids"][0]]["utr"]
            assert true_utr not in bank[m["bank_txn_id"]]["narration"]

    def test_missing_utr_has_no_reference_anywhere(self, data):
        truth = json.loads((data / "truth.json").read_text())
        pays = {p["payment_id"]: p for p in _read(data / "settlements.csv")}
        for m in truth["matches"]:
            if m["case"] != "missing_utr":
                continue
            for pid in m["payment_ids"]:
                assert pays[pid]["utr"] == ""

    def test_partial_batch_withheld_net_is_unique_in_group(self, data):
        """Without uniqueness two subsets tie, the matcher correctly reports
        ambiguity, and our own ground truth becomes unmatchable."""
        truth = json.loads((data / "truth.json").read_text())
        payments = _read(data / "settlements.csv")
        by_stl = {}
        for p in payments:
            by_stl.setdefault(p["settlement_id"], []).append(p)
        pay_by_id = {p["payment_id"]: p for p in payments}
        found = 0
        for m in truth["matches"]:
            if m["case"] != "partial_batch":
                continue
            found += 1
            withheld_net = int(pay_by_id[m["withheld_payment_id"]]["net_paise"])
            nets = [int(p["net_paise"]) for p in by_stl[m["settlement_id"]]]
            assert nets.count(withheld_net) == 1
            assert m["withheld_payment_id"] not in m["payment_ids"]
        assert found == 2

    def test_ambiguous_pair_is_a_genuine_tie(self, data):
        truth = json.loads((data / "truth.json").read_text())
        bank = {b["txn_id"]: b for b in _read(data / "bank.csv")}
        amb = [m for m in truth["matches"] if m["case"] == "ambiguous_amount"]
        assert len(amb) == 2
        a, b = (bank[m["bank_txn_id"]] for m in amb)
        assert a["credit_paise"] == b["credit_paise"]
        assert a["value_date"] == b["value_date"]

    def test_rounding_drift_exceeds_stage2_tolerance_sometimes(self, data):
        """Drift of 1-2 paise would be silently absorbed by a 2-paise
        tolerance and prove nothing; it must reach beyond that."""
        truth = json.loads((data / "truth.json").read_text())
        drifts = [abs(m["drift_paise"]) for m in truth["matches"]
                  if m["case"] == "rounding_drift"]
        assert drifts and all(1 <= d <= 5 for d in drifts)
        assert any(d > 2 for d in drifts), "no drift exceeds the 2-paise tolerance"

    def test_drift_never_breaks_the_payment_identity(self, data):
        """Drift lives on the bank credit, never inside a payment row."""
        for p in _read(data / "settlements.csv"):
            g, f, gs, n = (int(p["gross_paise"]), int(p["fee_paise"]),
                           int(p["gst_paise"]), int(p["net_paise"]))
            assert g - f - gs == n

    def test_foreign_credits_are_structurally_different(self, data):
        truth = json.loads((data / "truth.json").read_text())
        bank = {b["txn_id"]: b for b in _read(data / "bank.csv")}
        for u in truth["unmatchable_bank"]:
            narration = bank[u["bank_txn_id"]]["narration"]
            assert "RAZORPAY" not in narration.upper()


class TestStructure:
    def test_bank_balance_is_cumulative(self, data):
        truth = json.loads((data / "truth.json").read_text())
        running = truth["opening_balance_paise"]
        for row in _read(data / "bank.csv"):
            running += int(row["credit_paise"]) - int(row["debit_paise"])
            assert running == int(row["balance_paise"]), row["txn_id"]

    def test_unpaid_orders_have_no_payment(self, data):
        truth = json.loads((data / "truth.json").read_text())
        paid = {p["order_id"] for p in _read(data / "settlements.csv")}
        for u in truth["unpaid_orders"]:
            assert u["order_id"] not in paid

    def test_no_payment_claimed_twice(self, data):
        truth = json.loads((data / "truth.json").read_text())
        seen = set()
        for m in truth["matches"]:
            for pid in m["payment_ids"]:
                assert pid not in seen
                seen.add(pid)

    def test_every_bank_row_is_accounted_for(self, data):
        truth = json.loads((data / "truth.json").read_text())
        claimed = ({m["bank_txn_id"] for m in truth["matches"]}
                   | {u["bank_txn_id"] for u in truth["unmatchable_bank"]}
                   | {d["bank_txn_id"] for d in truth["debit_rows"]})
        assert {b["txn_id"] for b in _read(data / "bank.csv")} == claimed

    def test_clears_the_fifty_record_bar(self, data):
        counts = json.loads((data / "truth.json").read_text())["counts"]
        assert counts["orders"] >= 50
        assert counts["payments"] >= 50

    def test_narration_templates_rotate(self, data):
        """The regex must have real work to do, not one fixed format."""
        narrations = [b["narration"] for b in _read(data / "bank.csv")]
        shapes = {n.split("-")[0].split("/")[0].split()[0] for n in narrations}
        assert len(shapes) >= 3

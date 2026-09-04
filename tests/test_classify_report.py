"""Classification and report tests.

The exception list is the product, so these tests check that it is complete,
correctly categorised, and actionable -- not merely that a function returns.
"""

import pytest

from recon.classify import (
    CATEGORIES, classification_accuracy, classify_all, classify_deterministic,
)
from recon.generate import generate
from recon.ingest import load_dataset, load_truth
from recon.match_deterministic import run_deterministic
from recon.report import build_report, cash_position, git_commit
from recon.evaluate import evaluate


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    generate(42, 0.4, out)
    ds = load_dataset(out)
    truth = load_truth(out)
    result = run_deterministic(ds, stages=3)
    paid = {p.order_id for p in ds.payments}
    unpaid = {o.order_id for o in ds.orders if o.order_id not in paid}
    classified = classify_all(ds, result.unresolved, result.ambiguous, unpaid)
    return ds, truth, result, unpaid, classified


class TestClassification:
    def test_every_unresolved_record_is_classified(self, world):
        """Nothing may vanish between the matcher and the exception list."""
        ds, _, result, unpaid, classified = world
        ids = {c.exception.record_id for c in classified}
        for txn_id in result.unresolved:
            assert txn_id in ids, "%s unresolved but not on the list" % txn_id
        for order_id in unpaid:
            assert order_id in ids
        for txn in ds.debits:
            assert txn.txn_id in ids, "debit %s never surfaced" % txn.txn_id

    def test_categories_are_from_the_known_taxonomy(self, world):
        _, _, _, _, classified = world
        for c in classified:
            assert c.category in CATEGORIES

    def test_classification_matches_ground_truth(self, world):
        _, truth, _, _, classified = world
        acc = classification_accuracy(classified, truth)
        assert acc["scored"] > 0
        assert acc["accuracy"] == 1.0, acc["mistakes"]

    def test_reasons_are_actionable_prose(self, world):
        """A reason a finance analyst could act on, not a rule name."""
        _, _, _, _, classified = world
        for c in classified:
            assert len(c.reason) > 40, c.reason
            assert c.reason[0].isupper()
            assert c.reason.rstrip().endswith(".")

    def test_reasons_carry_the_amount(self, world):
        _, _, _, _, classified = world
        for c in classified:
            assert any(ch.isdigit() for ch in c.reason)

    def test_foreign_credits_are_named_as_non_gateway(self, world):
        _, truth, _, _, classified = world
        by_id = {c.exception.record_id: c for c in classified}
        for txn_id in truth.unmatchable_bank:
            assert by_id[txn_id].category == "foreign_credit"

    def test_ambiguous_reason_names_the_competing_batches(self, world):
        _, truth, _, _, classified = world
        by_id = {c.exception.record_id: c for c in classified}
        for tm in truth.matches:
            if str(tm.case) != "ambiguous_amount":
                continue
            c = by_id.get(tm.bank_txn_id)
            if c is None:
                continue
            assert "STL" in c.reason
            assert "will not guess" in c.reason

    def test_works_offline_with_no_proposer(self, world):
        """--no-llm must produce a complete, categorised exception list."""
        ds, _, result, unpaid, _ = world
        offline = classify_all(ds, result.unresolved, result.ambiguous,
                               unpaid, proposer=None)
        assert offline
        assert all(c.source in ("rule", "fallback") for c in offline)

    def test_unknown_bucket_is_used_rather_than_a_guess(self, world):
        """An unrecognisable record gets `unknown`, not a plausible label."""
        ds, _, _, _, _ = world
        out = classify_deterministic("NONEXISTENT", ds, "whatever")
        assert out is None


class TestCashPosition:
    def test_cash_position_reconciles(self, world):
        ds, _, result, _, _ = world
        cash = cash_position(ds, result.matches)
        assert cash["reconciles"]
        assert cash["derived_closing_paise"] == cash["closing_paise"]

    def test_opening_plus_flows_equals_closing(self, world):
        ds, _, result, _, _ = world
        cash = cash_position(ds, result.matches)
        assert (cash["opening_paise"] + cash["credits_paise"]
                - cash["debits_paise"]) == cash["closing_paise"]

    def test_fees_and_gst_are_reported(self, world):
        ds, _, result, _, _ = world
        cash = cash_position(ds, result.matches)
        assert cash["matched_fees_paise"] > 0
        assert cash["matched_gst_paise"] > 0


class TestReport:
    @pytest.fixture
    def ctx(self, world):
        ds, truth, result, unpaid, classified = world
        met = evaluate(result.matches, truth, ds, unpaid)
        return {
            "git_commit": git_commit(), "seed": 42, "hard_ratio": 0.4,
            "counts": {"orders": len(ds.orders), "payments": len(ds.payments),
                       "bank": len(ds.bank), "credits": len(ds.credits),
                       "debits": len(ds.debits), "settlements": 30},
            "elapsed": 0.05, "records": 418, "throughput": 8000.0,
            "metrics": met, "baseline_metrics": met,
            "ablation": [("Stages 1-3 (deterministic baseline)", met)],
            "exceptions": classified,
            "classification_accuracy": classification_accuracy(classified, truth),
            "cash": cash_position(ds, result.matches),
            "gate": None, "llm_stats": None, "llm_ran": False,
            "llm_description": "not used (--no-llm)", "sweep": None,
        }

    def test_all_eight_sections_render(self, ctx):
        md = build_report(ctx)
        for heading in ["## 1. Run header", "## 2. Headline metrics",
                        "## 3. Ablation", "## 4. Difficulty sweep",
                        "## 5. Verification gate", "## 6. Exception list",
                        "## 7. Cash position", "## 8. Throughput and cost"]:
            assert heading in md, "missing %s" % heading

    def test_exception_list_is_never_truncated(self, ctx):
        """SPEC: section 6 must be complete. Every record, every time."""
        md = build_report(ctx)
        for c in ctx["exceptions"]:
            assert c.exception.record_id in md
        assert "..." not in md.split("## 6.")[1].split("## 7.")[0]

    def test_forced_match_errors_are_prominent_when_nonzero(self, ctx):
        ctx["metrics"].forced_match_errors = 2
        md = build_report(ctx)
        assert "most important number" in md

    def test_zero_forced_errors_is_stated_plainly(self, ctx):
        md = build_report(ctx)
        assert "corresponds to nothing" in md

    def test_amounts_are_rupees_not_paise(self, ctx):
        md = build_report(ctx)
        assert "Rs " in md
        # A raw paise integer for the opening balance would be 5000000.
        assert "5000000" not in md

    def test_report_is_valid_markdown_tables(self, ctx):
        md = build_report(ctx)
        for line in md.split("\n"):
            if line.startswith("|") and "---" not in line:
                assert line.rstrip().endswith("|"), line

    def test_git_commit_never_crashes(self):
        assert isinstance(git_commit(), str)
        assert git_commit()

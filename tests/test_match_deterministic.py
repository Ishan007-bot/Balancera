"""Deterministic matcher tests.

SPEC section 14 priority 3: subset-sum with known inputs and known outputs,
including the no-solution and multiple-solution cases.

The tests that matter most here are the ones asserting the matcher *refuses*
to match. A matcher that guesses scores better on a naive match rate and is
useless in production; these lock in the refusals.
"""

import pytest

from recon.ingest import load_dataset, load_truth
from recon.match_deterministic import (
    MatchResult, canonical_ref, extract_reference, find_subsets,
    run_deterministic, stage1_reference, stage2_group_sum, stage3_subset_sum,
)
from recon.generate import generate


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    generate(42, 0.4, out)
    return load_dataset(out), load_truth(out)


class TestSubsetSum:
    """Known inputs, known outputs -- including no-solution and multi-solution."""

    def test_exact_single_solution(self):
        found = find_subsets([100, 200, 300, 400], 500, tolerance=0)
        assert found
        for sub in found:
            assert sum([100, 200, 300, 400][i] for i in sub) == 500

    def test_full_set_is_a_valid_subset(self):
        found = find_subsets([100, 200, 300], 600, tolerance=0)
        assert (0, 1, 2) in found

    def test_no_solution(self):
        assert find_subsets([100, 200, 400], 50, tolerance=0) == []
        assert find_subsets([100, 200, 400], 10_000, tolerance=0) == []

    def test_multiple_solutions_are_detected(self):
        """Two distinct subsets summing to the target must both be found, so
        the caller can refuse rather than pick one."""
        found = find_subsets([100, 200, 300, 100, 200], 300, tolerance=0, limit=2)
        assert len(found) >= 2

    def test_tolerance_absorbs_drift(self):
        assert find_subsets([100, 200], 301, tolerance=2)
        assert find_subsets([100, 200], 303, tolerance=2) == []

    def test_empty_subset_is_never_returned(self):
        """Target 0 must not match by selecting nothing."""
        assert find_subsets([100, 200], 0, tolerance=0) == []

    def test_empty_input(self):
        assert find_subsets([], 100, tolerance=0) == []

    def test_odd_length_splits_correctly(self):
        """Meet-in-the-middle on an odd-length list must not drop an element."""
        amounts = [1, 2, 4, 8, 16, 32, 64]
        found = find_subsets(amounts, 127, tolerance=0)
        assert (0, 1, 2, 3, 4, 5, 6) in found

    def test_bounded_group_size_stays_fast(self):
        """15 items is the cap; this must complete effectively instantly."""
        import time
        amounts = list(range(1000, 1000 + 15))
        start = time.perf_counter()
        find_subsets(amounts, sum(amounts) - 1007, tolerance=0)
        assert time.perf_counter() - start < 0.5


class TestReferenceExtraction:
    def test_each_narration_template(self):
        assert extract_reference("NEFT-RAZORPAYSOFT-UTR8842910-SETTLEMENT")
        assert extract_reference("IMPS/UTR3592974/RAZORPAY/COLLECTION")
        assert extract_reference("RTGS RZPYSOFT UTR6120253 NET STLMT")
        assert extract_reference("UPI-RAZORPAY-UTR8795419")
        assert extract_reference("UTR8973915 RAZORPAYSOFTWA")

    def test_foreign_narrations_yield_nothing(self):
        assert extract_reference("NEFT-ACMESUPPLIES-INV4471") is None
        assert extract_reference("NEFT-VENDORREFUND-BILL2290") is None

    def test_canonical_ref_ignores_prefix(self):
        """Narration and settlements disagree about the UTR prefix; comparing
        digits is what makes the two forms equal."""
        assert canonical_ref("UTR8973915") == canonical_ref("8973915")
        assert canonical_ref("UTR8973915") == "8973915"


class TestStageBehaviour:
    def test_stage1_rejects_reference_hit_with_wrong_amount(self, ds):
        """A reference match with a disagreeing amount is a data problem, not
        a match. Accepting it is the failure mode this project prevents."""
        dataset, truth = ds
        result = MatchResult()
        stage1_reference(dataset, result)
        for txn_id, why in result.unresolved.items():
            if "amounts disagree" in why:
                assert txn_id not in result.matched_txn_ids
                break
        else:
            pytest.skip("no amount-disagreement case in this dataset")

    def test_stage2_refuses_a_genuine_tie(self, ds):
        """Two settlement groups with the same total and date must leave the
        credit unresolved, with both candidates recorded."""
        dataset, truth = ds
        result = MatchResult()
        stage2_group_sum(dataset, result)
        amb_txns = {m.bank_txn_id for m in truth.matches
                    if str(m.case) == "ambiguous_amount"}
        assert amb_txns, "dataset has no ambiguous pair"
        for txn_id in amb_txns:
            assert txn_id not in result.matched_txn_ids, \
                "stage 2 guessed on a genuine tie"
            assert len(result.ambiguous.get(txn_id, [])) >= 2

    def test_stage3_resolves_partial_batches(self, ds):
        dataset, truth = ds
        result = run_deterministic(dataset, stages=3)
        matched = {m.bank_txn_id: m for m in result.matches}
        for tm in truth.matches:
            if str(tm.case) != "partial_batch":
                continue
            assert tm.bank_txn_id in matched, "partial_batch not resolved"
            assert matched[tm.bank_txn_id].payment_ids == tm.payment_ids

    def test_no_payment_is_claimed_twice(self, ds):
        dataset, _ = ds
        result = run_deterministic(dataset, stages=3)
        seen = set()
        for m in result.matches:
            assert not (seen & m.payment_ids), "payment claimed by two matches"
            seen |= set(m.payment_ids)

    def test_resolved_credits_carry_no_stale_exception(self, ds):
        """An earlier stage's failure reason must not survive a later stage's
        success, or resolved rows appear on the exception list."""
        dataset, _ = ds
        result = run_deterministic(dataset, stages=3)
        assert not (result.matched_txn_ids & set(result.unresolved))
        assert not (result.matched_txn_ids & set(result.ambiguous))


class TestSafetyProperties:
    """The properties that make this system trustworthy, not just accurate."""

    def test_foreign_credits_are_never_matched(self, ds):
        """The single most important assertion in the suite."""
        dataset, truth = ds
        result = run_deterministic(dataset, stages=3)
        matched = result.matched_txn_ids
        for txn_id in truth.unmatchable_bank:
            assert txn_id not in matched, \
                "forced a match on %s, which corresponds to nothing" % txn_id

    def test_debit_rows_are_never_matched(self, ds):
        dataset, truth = ds
        result = run_deterministic(dataset, stages=3)
        for txn_id in truth.debit_rows:
            assert txn_id not in result.matched_txn_ids

    def test_every_match_is_arithmetically_sound(self, ds):
        """Recompute every accepted match from source amounts."""
        dataset, _ = ds
        result = run_deterministic(dataset, stages=3)
        for m in result.matches:
            credit = dataset.bank_by_id[m.bank_txn_id].credit_paise
            total = dataset.net_sum(m.payment_ids)
            assert abs(total - credit) <= 2, \
                "%s: payments sum %d vs credit %d" % (m.bank_txn_id, total, credit)

    def test_every_credit_is_matched_or_explained(self, ds):
        """Nothing may disappear silently -- the exception list is the product."""
        dataset, _ = ds
        result = run_deterministic(dataset, stages=3)
        for txn in dataset.credits:
            assert (txn.txn_id in result.matched_txn_ids
                    or txn.txn_id in result.unresolved), \
                "%s vanished: neither matched nor explained" % txn.txn_id

    def test_stages_are_monotonic(self, ds):
        """More stages must never match fewer transactions."""
        dataset, _ = ds
        counts = [len(run_deterministic(dataset, stages=n).matches)
                  for n in (1, 2, 3)]
        assert counts == sorted(counts)


class TestDeterminism:
    def test_repeated_runs_are_identical(self, ds):
        dataset, _ = ds
        a = run_deterministic(dataset, stages=3)
        b = run_deterministic(dataset, stages=3)
        assert sorted(m.key() for m in a.matches) == \
               sorted(m.key() for m in b.matches)
        assert a.unresolved == b.unresolved

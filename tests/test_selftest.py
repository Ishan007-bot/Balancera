"""Adversarial self-test tests.

The self-test is the live demonstration that the gate does real work, so it
needs its own guarantee: that it genuinely corrupts things, genuinely catches
them, and would actually fail if the gate stopped working.
"""

import pytest

from recon.generate import generate
from recon.ingest import load_dataset
from recon.match_deterministic import run_deterministic
from recon.selftest import build_attacks, format_selftest, run_selftest
from recon.verify import Rule


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    generate(42, 0.4, out)
    ds = load_dataset(out)
    result = run_deterministic(ds, stages=3)
    return ds, result


def test_every_corruption_is_caught(world):
    ds, result = world
    results, all_caught = run_selftest(ds, result.matches,
                                       result.claimed_payment_ids)
    assert results, "no attacks were generated"
    assert all_caught, [r for r in results if not r["caught"]]


def test_attacks_cover_the_important_rules(world):
    ds, result = world
    attacks = build_attacks(ds, result.matches, result.claimed_payment_ids)
    covered = {a.expected_rule for a in attacks}
    for rule in (Rule.AMOUNT_MISMATCH, Rule.UNKNOWN_PAYMENT,
                 Rule.ALREADY_CLAIMED, Rule.DATE_WINDOW, Rule.NOT_A_CREDIT,
                 Rule.LOW_CONFIDENCE):
        assert rule in covered, "no attack exercises %s" % rule


def test_each_attack_fails_for_its_own_reason(world):
    """An attack that trips a different rule is not testing what it claims."""
    ds, result = world
    results, _ = run_selftest(ds, result.matches, result.claimed_payment_ids)
    for r in results:
        assert r["actual_rule"] == r["expected_rule"], r


def test_the_selftest_leaves_the_dataset_unchanged(world):
    """Doctored bank rows must be restored, or later stages see bad data."""
    ds, result = world
    before = {t.txn_id: (t.value_date, t.credit_paise) for t in ds.bank}
    run_selftest(ds, result.matches, result.claimed_payment_ids)
    after = {tid: (ds.bank_by_id[tid].value_date,
                   ds.bank_by_id[tid].credit_paise) for tid in before}
    assert before == after


def test_selftest_would_fail_if_the_gate_stopped_working(world, monkeypatch):
    """The self-test must be capable of failing -- one that always passes
    proves nothing about the gate."""
    ds, result = world
    import recon.selftest as st

    class AlwaysAccepts:
        accepted = True
        failed_rule = None
        reason = "rubber stamp"

    monkeypatch.setattr(st, "verify", lambda *a, **k: AlwaysAccepts())
    _, all_caught = st.run_selftest(ds, result.matches,
                                    result.claimed_payment_ids)
    assert not all_caught


def test_output_names_the_rule_and_the_numbers(world):
    ds, result = world
    results, _ = run_selftest(ds, result.matches, result.claimed_payment_ids)
    text = format_selftest(results)
    assert "CAUGHT" in text
    assert "amount_mismatch" in text
    assert "corruptions caught" in text


def test_no_attacks_when_there_are_no_matches(world):
    ds, _ = world
    assert build_attacks(ds, [], set()) == []

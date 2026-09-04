"""LLM proposal layer tests.

Every test mocks the model. No test may require an API key, and none does --
the module is even importable with the anthropic package absent, because the
offline path is a first-class requirement rather than a fallback.
"""

import json

import pytest

from recon.generate import generate
from recon.ingest import load_dataset, load_truth
from recon.match_deterministic import run_deterministic
from recon.match_llm import (
    ACTIONS, DEFAULT_PROVIDER, LLMProposer, Proposal, build_prompt,
    parse_response, prompt_hash, strip_fences, SYSTEM_PROMPT,
)
from recon.retrieve import retrieve

from .mock_llm import (
    MockClient, always_proposes_everything, asks_to_widen_then_abstains,
    group_aware_solver, hallucinates_payment_ids, honest_solver,
    returns_prose, wrapped_in_markdown_fences,
)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    generate(42, 0.4, out)
    ds = load_dataset(out)
    truth = load_truth(out)
    result = run_deterministic(ds, stages=3)
    residual = [t for t in ds.credits if t.txn_id not in result.matched_txn_ids]
    return ds, truth, result, residual


def make(tmp_path, **kwargs):
    return LLMProposer(cache_dir=tmp_path / "cache",
                       log_path=tmp_path / "llm_calls.jsonl", **kwargs)


class TestResponseParsing:
    def test_plain_json(self):
        p = parse_response(json.dumps({
            "action": "propose_match", "proposed_payment_ids": ["PAY1"],
            "confidence": 0.9, "reasoning": "why", "abstain": False}), "BNK1")
        assert p.proposed_payment_ids == ["PAY1"]
        assert p.confidence == 0.9
        assert not p.abstain

    def test_markdown_fences_are_stripped(self):
        """The prompt forbids fences; parsing must not rely on that."""
        raw = '```json\n{"proposed_payment_ids": ["PAY1"], "confidence": 0.8}\n```'
        assert parse_response(raw, "BNK1").proposed_payment_ids == ["PAY1"]

    def test_prose_around_json_is_tolerated(self):
        raw = 'Here is my answer:\n{"proposed_payment_ids": ["PAY1"], "confidence": 0.5}\nHope that helps.'
        assert parse_response(raw, "BNK1").proposed_payment_ids == ["PAY1"]

    def test_unparseable_response_becomes_an_abstain(self):
        p = parse_response("I think maybe these ones?", "BNK1")
        assert p.abstain
        assert p.error
        assert not p.is_actionable()

    def test_non_object_json_is_rejected(self):
        assert parse_response("[1, 2, 3]", "BNK1").abstain

    def test_malformed_payment_ids_rejected(self):
        p = parse_response('{"proposed_payment_ids": [1, 2]}', "BNK1")
        assert p.abstain and p.error

    def test_confidence_is_clamped(self):
        assert parse_response('{"confidence": 5.0}', "BNK1").confidence == 1.0
        assert parse_response('{"confidence": -2}', "BNK1").confidence == 0.0
        assert parse_response('{"confidence": "junk"}', "BNK1").confidence == 0.0

    def test_unknown_action_falls_back(self):
        p = parse_response('{"action": "delete_everything"}', "BNK1")
        assert p.action in ACTIONS

    def test_duplicate_ids_are_collapsed(self):
        p = parse_response(
            '{"proposed_payment_ids": ["PAY1", "PAY1", "PAY2"]}', "BNK1")
        assert p.proposed_payment_ids == ["PAY1", "PAY2"]

    def test_abstain_action_sets_the_flag(self):
        assert parse_response('{"action": "abstain"}', "BNK1").abstain

    def test_strip_fences_leaves_clean_json_alone(self):
        assert strip_fences('{"a": 1}') == '{"a": 1}'


class TestPromptConstruction:
    def test_prompt_contains_the_credit_and_candidates(self, world):
        ds, _, result, residual = world
        cs = retrieve(ds, residual[1], result.claimed_payment_ids)
        prompt = build_prompt(cs, ds)
        assert residual[1].txn_id in prompt
        assert str(residual[1].credit_paise) in prompt
        for c in cs.candidates:
            assert c.payment.payment_id in prompt

    def test_prompt_is_deterministic(self, world):
        """Same inputs must give identical bytes or the cache never hits."""
        ds, _, result, residual = world
        cs = retrieve(ds, residual[0], result.claimed_payment_ids)
        assert build_prompt(cs, ds) == build_prompt(cs, ds)

    def test_hash_is_stable_and_input_sensitive(self, world):
        ds, _, result, residual = world
        a = build_prompt(retrieve(ds, residual[0], result.claimed_payment_ids), ds)
        b = build_prompt(retrieve(ds, residual[1], result.claimed_payment_ids), ds)
        assert prompt_hash(SYSTEM_PROMPT, a) == prompt_hash(SYSTEM_PROMPT, a)
        assert prompt_hash(SYSTEM_PROMPT, a) != prompt_hash(SYSTEM_PROMPT, b)

    def test_system_prompt_encourages_abstention(self):
        """A model that always answers is worse than useless here."""
        assert "ABSTAIN" in SYSTEM_PROMPT
        assert "valued outcome" in SYSTEM_PROMPT


class TestRetrieval:
    def test_model_never_sees_the_whole_dataset(self, world):
        ds, _, result, residual = world
        for txn in residual:
            cs = retrieve(ds, txn, result.claimed_payment_ids, k=20)
            assert len(cs.candidates) <= 20
            assert len(cs.candidates) < len(ds.payments)

    def test_claimed_payments_are_never_offered(self, world):
        ds, _, result, residual = world
        claimed = result.claimed_payment_ids
        for txn in residual:
            assert not (retrieve(ds, txn, claimed).payment_ids & claimed)

    def test_retrieval_is_deterministic(self, world):
        ds, _, result, residual = world
        a = retrieve(ds, residual[0], result.claimed_payment_ids)
        b = retrieve(ds, residual[0], result.claimed_payment_ids)
        assert [c.payment.payment_id for c in a.candidates] == \
               [c.payment.payment_id for c in b.candidates]

    def test_the_correct_answer_is_actually_offered(self, world):
        """Retrieval must not hide the truth from the model -- otherwise a
        low match rate measures retrieval, not reasoning."""
        ds, truth, result, residual = world
        for txn in residual:
            tm = truth.by_txn.get(txn.txn_id)
            if tm is None:
                continue
            offered = retrieve(ds, txn, result.claimed_payment_ids).payment_ids
            assert tm.payment_ids <= offered, \
                "%s: truth payments were not offered" % txn.txn_id


class TestProposalFlow:
    def test_honest_model_solves_the_rounding_drift_case(self, world, tmp_path):
        ds, truth, result, residual = world
        prop = make(tmp_path, client=MockClient(behaviour=group_aware_solver()))
        target = next(t for t in residual
                      if t.txn_id in truth.by_txn
                      and str(truth.by_txn[t.txn_id].case) == "rounding_drift")
        p = prop.propose(ds, target, result.claimed_payment_ids)
        assert p.is_actionable()
        assert set(p.proposed_payment_ids) == set(truth.by_txn[target.txn_id].payment_ids)

    def test_model_abstains_on_a_genuine_tie(self, world, tmp_path):
        """The ambiguous pair must provoke a refusal, not a coin flip."""
        ds, truth, result, residual = world
        prop = make(tmp_path, client=MockClient(behaviour=group_aware_solver()))
        for txn in residual:
            tm = truth.by_txn.get(txn.txn_id)
            if tm and str(tm.case) == "ambiguous_amount":
                p = prop.propose(ds, txn, result.claimed_payment_ids)
                assert p.abstain, "model guessed on a genuine tie"

    def test_model_abstains_on_foreign_credits(self, world, tmp_path):
        ds, truth, result, residual = world
        prop = make(tmp_path, client=MockClient(behaviour=group_aware_solver()))
        for txn in residual:
            if txn.txn_id in truth.unmatchable_bank:
                p = prop.propose(ds, txn, result.claimed_payment_ids)
                assert p.abstain, "%s matches nothing but was proposed" % txn.txn_id

    def test_prose_response_becomes_an_abstain(self, world, tmp_path):
        ds, _, result, residual = world
        prop = make(tmp_path, client=MockClient(behaviour=returns_prose))
        p = prop.propose(ds, residual[1], result.claimed_payment_ids)
        assert p.abstain and not p.is_actionable()

    def test_fenced_response_is_still_parsed(self, world, tmp_path):
        ds, _, result, residual = world
        prop = make(tmp_path, client=MockClient(behaviour=wrapped_in_markdown_fences))
        p = prop.propose(ds, residual[1], result.claimed_payment_ids)
        assert p.error is None

    def test_client_failure_becomes_an_abstain_not_a_crash(self, world, tmp_path):
        ds, _, result, residual = world

        class Boom:
            class messages:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("network down")

        prop = make(tmp_path, client=Boom())
        p = prop.propose(ds, residual[1], result.claimed_payment_ids)
        assert p.abstain and "llm call failed" in p.error


class TestAgenticLoop:
    def test_widen_request_triggers_another_iteration(self, world, tmp_path):
        ds, _, result, residual = world
        client = MockClient(behaviour=asks_to_widen_then_abstains)
        prop = make(tmp_path, client=client)
        p = prop.propose(ds, residual[1], result.claimed_payment_ids)
        assert p.iterations == 2
        assert len(client.calls) == 2
        assert p.abstain

    def test_loop_is_bounded(self, world, tmp_path):
        """A model that keeps asking must not loop forever."""
        ds, _, result, residual = world

        def always_widen(user_prompt, i):
            return json.dumps({"action": "widen_date_window",
                               "proposed_payment_ids": [], "confidence": 0.0,
                               "abstain": False, "reasoning": "again"})

        client = MockClient(behaviour=always_widen)
        prop = make(tmp_path, client=client, max_iterations=3)
        p = prop.propose(ds, residual[1], result.claimed_payment_ids)
        assert len(client.calls) <= 3
        assert p.iterations <= 3

    def test_single_iteration_when_model_answers_immediately(self, world, tmp_path):
        ds, _, result, residual = world
        client = MockClient(behaviour=group_aware_solver())
        prop = make(tmp_path, client=client)
        p = prop.propose(ds, residual[1], result.claimed_payment_ids)
        assert p.iterations == 1


class TestCachingAndAudit:
    def test_second_run_makes_no_api_calls(self, world, tmp_path):
        ds, _, result, residual = world
        client = MockClient(behaviour=group_aware_solver())
        prop = make(tmp_path, client=client)
        for txn in residual:
            prop.propose(ds, txn, result.claimed_payment_ids)
        first = len(client.calls)
        for txn in residual:
            prop.propose(ds, txn, result.claimed_payment_ids)
        assert len(client.calls) == first, "cache did not prevent re-calling"
        assert prop.cache_hits > 0

    def test_cached_results_are_identical(self, world, tmp_path):
        ds, _, result, residual = world
        prop = make(tmp_path, client=MockClient(behaviour=group_aware_solver()))
        a = prop.propose(ds, residual[1], result.claimed_payment_ids)
        b = prop.propose(ds, residual[1], result.claimed_payment_ids)
        assert a.proposed_payment_ids == b.proposed_payment_ids
        assert b.from_cache

    def test_corrupt_cache_entry_is_a_miss_not_a_crash(self, world, tmp_path):
        ds, _, result, residual = world
        prop = make(tmp_path, client=MockClient(behaviour=group_aware_solver()))
        prop.propose(ds, residual[1], result.claimed_payment_ids)
        for path in (tmp_path / "cache").glob("*.json"):
            path.write_text("{ not json")
        assert prop.propose(ds, residual[1], result.claimed_payment_ids) is not None

    def test_every_call_is_logged(self, world, tmp_path):
        ds, _, result, residual = world
        prop = make(tmp_path, client=MockClient(behaviour=group_aware_solver()))
        for txn in residual:
            prop.propose(ds, txn, result.claimed_payment_ids)
        lines = (tmp_path / "llm_calls.jsonl").read_text().strip().split("\n")
        assert lines and lines[0]
        for line in lines:
            entry = json.loads(line)
            for field in ("bank_txn_id", "prompt_hash", "candidates",
                          "raw_response", "input_tokens", "output_tokens"):
                assert field in entry, "audit log missing %s" % field

    def test_stats_report_tokens_and_cost(self, world, tmp_path):
        ds, _, result, residual = world
        prop = make(tmp_path, client=MockClient(behaviour=group_aware_solver()))
        for txn in residual:
            prop.propose(ds, txn, result.claimed_payment_ids)
        stats = prop.stats()
        assert stats["llm_calls"] > 0
        assert stats["input_tokens"] > 0
        assert stats["output_tokens"] > 0
        assert stats["model"]  # whichever model/provider ran
        assert stats["provider"] == "mock"
        # The mock model has no price entry, so cost is legitimately zero.
        # Cost accounting itself is checked against a real priced model below.
        assert stats["estimated_cost_usd"] == 0.0

    def test_cost_is_computed_from_provider_pricing(self):
        from recon.providers import PRICING

        prop = LLMProposer(cache_dir="runs/cache", client=MockClient())
        prop._client = None
        from recon.providers import get_provider
        prop._provider = get_provider("anthropic")
        in_rate, out_rate = PRICING["claude-sonnet-5"]
        expected = round(1e6 / 1e6 * in_rate + 1e6 / 1e6 * out_rate, 6)
        assert prop._cost(1_000_000, 1_000_000) == expected


class TestDangerousModels:
    """The proposer must not launder a bad model into a good answer. These
    assert only that the proposal survives to the gate -- verify.py is what
    rejects it, and it is tested separately in Phase 4."""

    def test_overconfident_proposal_is_still_only_a_proposal(self, world, tmp_path):
        ds, _, result, residual = world
        prop = make(tmp_path, client=MockClient(behaviour=always_proposes_everything))
        p = prop.propose(ds, residual[1], result.claimed_payment_ids)
        assert isinstance(p, Proposal)
        assert p.confidence == 0.99  # passed through unaltered, not trusted

    def test_hallucinated_ids_are_passed_through_for_the_gate(self, world, tmp_path):
        ds, _, result, residual = world
        prop = make(tmp_path, client=MockClient(behaviour=hallucinates_payment_ids))
        p = prop.propose(ds, residual[1], result.claimed_payment_ids)
        assert p.proposed_payment_ids == ["PAY99991", "PAY99992"]
        assert not any(pid in ds.payments_by_id for pid in p.proposed_payment_ids)


def test_module_imports_without_the_sdk():
    """--no-llm must work on a machine with no provider SDK installed."""
    import importlib
    import recon.match_llm as m
    importlib.reload(m)
    assert m.DEFAULT_PROVIDER == "anthropic"


def test_provider_registry_lists_the_free_options():
    from recon.providers import PROVIDERS
    assert set(PROVIDERS) == {"anthropic", "groq", "gemini"}
    for name, (model, env_var) in PROVIDERS.items():
        assert model and env_var.endswith("_API_KEY")


def test_unavailable_provider_reports_why_not():
    """A missing key or SDK must produce a clear message, not a crash."""
    from recon.providers import get_provider
    ok, why = get_provider("groq", api_key=None).available()
    if not ok:
        assert "GROQ_API_KEY" in why or "openai package" in why


class TestFailedCallsAreNotAbstentions:
    """Regression: a failing API call returned abstain with an empty reason,
    so a broken run was indistinguishable from a cautious model. Abstention
    rate is a reported metric -- it must never count failures."""

    def test_failure_is_flagged_in_reasoning_not_just_error(self, world, tmp_path):
        ds, _, result, residual = world

        class Failing:
            class messages:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("401 Invalid API Key")

        prop = make(tmp_path, client=Failing())
        p = prop.propose(ds, residual[1], result.claimed_payment_ids)
        assert p.error, "failure not recorded"
        assert "FAILED" in p.reasoning, \
            "a failed call must be visible wherever the proposal is displayed"
        assert "Invalid API Key" in p.reasoning

    def test_failed_calls_are_counted_separately(self, world, tmp_path):
        ds, _, result, residual = world

        class Failing:
            class messages:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("network down")

        prop = make(tmp_path, client=Failing())
        proposals = [prop.propose(ds, t, result.claimed_payment_ids)
                     for t in residual]
        # A credit with no candidates in the window abstains before any call
        # is attempted, so failures count the calls actually tried -- not
        # every residual credit.
        attempted = sum(1 for p in proposals if p.error)
        assert prop.stats()["failed_calls"] == attempted
        assert attempted > 0
        assert prop.stats()["llm_calls"] == 0, "a failed call is not a call"

    def test_a_genuine_abstention_has_no_error(self, world, tmp_path):
        ds, _, result, residual = world
        prop = make(tmp_path, client=MockClient(behaviour=group_aware_solver()))
        p = prop.propose(ds, residual[0], result.claimed_payment_ids)
        assert p.abstain
        assert p.error is None, "a real abstention must not look like a failure"

"""A scriptable fake Anthropic client.

Every test mocks the LLM -- SPEC section 14 is explicit that no test may
require an API key. This mock mimics the response shape the real SDK returns
(``.content[].text`` and ``.usage``) closely enough that ``match_llm`` cannot
tell the difference, so the code under test is the real code path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class _Block:
    text: str
    type: str = "text"


@dataclass
class _Usage:
    input_tokens: int = 500
    output_tokens: int = 80


@dataclass
class _Response:
    content: list
    usage: _Usage


class _Messages:
    def __init__(self, owner):
        self._owner = owner

    def create(self, **kwargs):
        return self._owner._respond(**kwargs)


class MockClient:
    """Returns scripted responses; records every call for assertions."""

    def __init__(self, responses=None, behaviour=None):
        #: list of raw strings to return in order, or a callable(user_prompt)
        self.responses = list(responses or [])
        self.behaviour = behaviour
        self.calls: list[dict] = []
        self.messages = _Messages(self)

    def _respond(self, **kwargs):
        user = kwargs["messages"][0]["content"]
        self.calls.append(kwargs)

        if self.behaviour is not None:
            text = self.behaviour(user, len(self.calls) - 1)
        elif self.responses:
            text = self.responses.pop(0)
        else:
            text = json.dumps({"action": "abstain", "proposed_payment_ids": [],
                               "confidence": 0.0, "abstain": True,
                               "reasoning": "no scripted response"})
        return _Response(content=[_Block(text=text)], usage=_Usage())


# -- behaviours ------------------------------------------------------------

def parse_candidates(user_prompt: str) -> list[tuple[str, int]]:
    """Pull (payment_id, net_paise) pairs out of a rendered prompt."""
    return [(m.group(1), int(m.group(2))) for m in
            re.finditer(r"(PAY\d+)\s+net\s+(\d+) paise", user_prompt)]


def parse_credit(user_prompt: str) -> int:
    m = re.search(r"amount\s+:\s+(\d+) paise", user_prompt)
    return int(m.group(1)) if m else 0


def honest_solver(tolerance: int = 6):
    """A competent model: solves what it can, abstains otherwise.

    Tries the full candidate set, then every leave-one-out subset -- enough to
    resolve group sums and partial batches without becoming a second matcher.
    """
    def behave(user_prompt, call_index):
        credit = parse_credit(user_prompt)
        cands = parse_candidates(user_prompt)
        total = sum(n for _, n in cands)

        if abs(total - credit) <= tolerance:
            return json.dumps({
                "action": "propose_match",
                "proposed_payment_ids": [p for p, _ in cands],
                "confidence": 0.93,
                "reasoning": "the %d candidate payments sum to %d against a "
                             "credit of %d" % (len(cands), total, credit),
                "abstain": False,
            })
        for pid, net in cands:
            if abs(total - net - credit) <= tolerance:
                return json.dumps({
                    "action": "propose_match",
                    "proposed_payment_ids": [p for p, _ in cands if p != pid],
                    "confidence": 0.87,
                    "reasoning": "excluding %s (net %d) the remainder sums to "
                                 "%d against credit %d"
                                 % (pid, net, total - net, credit),
                    "abstain": False,
                })
        return json.dumps({
            "action": "abstain", "proposed_payment_ids": [],
            "confidence": 0.0, "abstain": True,
            "reasoning": "no subset of the candidates sums to %d" % credit,
        })
    return behave


def parse_settlement_groups(user_prompt: str) -> dict[str, list[tuple[str, int]]]:
    """Group the offered candidates by settlement, as the prompt renders them."""
    groups: dict[str, list[tuple[str, int]]] = {}
    current = None
    for line in user_prompt.split("\n"):
        header = re.search(r"settlement (STL\d+) --", line)
        if header:
            current = header.group(1)
            groups[current] = []
            continue
        row = re.search(r"(PAY\d+)\s+net\s+(\d+) paise", line)
        if row and current:
            groups[current].append((row.group(1), int(row.group(2))))
    return groups


def group_aware_solver(tolerance: int = 6):
    """A more capable model: reasons per settlement group.

    Matches a whole group, or a group minus one payment. Critically, when two
    different groups both fit, it abstains -- which is the behaviour a genuine
    tie should provoke and the reason the ambiguous pair exists.
    """
    def behave(user_prompt, call_index):
        credit = parse_credit(user_prompt)
        groups = parse_settlement_groups(user_prompt)

        solutions = []
        for stl, members in groups.items():
            total = sum(n for _, n in members)
            if abs(total - credit) <= tolerance:
                solutions.append((stl, [p for p, _ in members], total,
                                  "settlement %s sums to %d" % (stl, total)))
                continue
            for pid, net in members:
                if abs(total - net - credit) <= tolerance:
                    solutions.append((
                        stl, [p for p, _ in members if p != pid], total - net,
                        "settlement %s minus %s sums to %d"
                        % (stl, pid, total - net)))
                    break

        if len(solutions) == 1:
            stl, ids, total, why = solutions[0]
            return json.dumps({
                "action": "propose_match", "proposed_payment_ids": ids,
                "confidence": 0.91,
                "reasoning": "%s against a credit of %d" % (why, credit),
                "abstain": False,
            })
        if len(solutions) > 1:
            return json.dumps({
                "action": "abstain", "proposed_payment_ids": [],
                "confidence": 0.0, "abstain": True,
                "reasoning": "%d settlements each sum to %d (%s) -- no basis "
                             "to choose between them"
                             % (len(solutions), credit,
                                ", ".join(s[0] for s in solutions)),
            })
        return json.dumps({
            "action": "abstain", "proposed_payment_ids": [],
            "confidence": 0.0, "abstain": True,
            "reasoning": "no settlement group sums to %d" % credit,
        })
    return behave


def always_proposes_everything(user_prompt, call_index):
    """A dangerous model: never abstains, always names every candidate.

    Used to prove the verification gate rejects confident nonsense.
    """
    cands = parse_candidates(user_prompt)
    return json.dumps({
        "action": "propose_match",
        "proposed_payment_ids": [p for p, _ in cands],
        "confidence": 0.99,
        "reasoning": "these look right to me",
        "abstain": False,
    })


def hallucinates_payment_ids(user_prompt, call_index):
    """Proposes ids that were never offered -- the gate must catch this."""
    return json.dumps({
        "action": "propose_match",
        "proposed_payment_ids": ["PAY99991", "PAY99992"],
        "confidence": 0.95,
        "reasoning": "invented ids",
        "abstain": False,
    })


def asks_to_widen_then_abstains(user_prompt, call_index):
    """Exercises the agentic loop: one action, then a decision."""
    if call_index == 0:
        return json.dumps({
            "action": "widen_date_window", "proposed_payment_ids": [],
            "confidence": 0.0, "abstain": False,
            "reasoning": "candidates look close but the dates are outside "
                         "the window",
        })
    return json.dumps({
        "action": "abstain", "proposed_payment_ids": [],
        "confidence": 0.0, "abstain": True,
        "reasoning": "still nothing defensible after widening",
    })


def wrapped_in_markdown_fences(user_prompt, call_index):
    """The prompt forbids fences; parsing must not depend on that holding."""
    return ('```json\n' + json.dumps({
        "action": "abstain", "proposed_payment_ids": [], "confidence": 0.0,
        "abstain": True, "reasoning": "fenced anyway"}) + '\n```')


def returns_prose(user_prompt, call_index):
    return "I think these payments probably match, but I am not certain."

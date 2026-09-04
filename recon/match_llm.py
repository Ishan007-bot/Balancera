"""LLM proposal layer.

This module may only *propose*. Nothing it returns becomes a financial fact:
every proposal is re-checked from raw source data by ``verify.py``, which
imports nothing from here and knows nothing about models.

Three properties matter more than accuracy here:

* **Abstention is first-class.** ``abstain: true`` is explicitly encouraged in
  the prompt. A model that always answers is worse than useless on a task
  where the right answer is often "these do not correspond".
* **Everything is cached and logged.** Responses are keyed by a hash of the
  prompt, so re-runs are free and reproducible; every call is appended to a
  JSONL audit trail with candidates, response, latency and cost.
* **Offline is a real code path.** With ``--no-llm`` (or no SDK, or no key)
  this module is never constructed and the pipeline still completes.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .ingest import Dataset
from .models import BankTxn
from .money import format_paise
from .retrieve import DEFAULT_K, CandidateSet, retrieve

MODEL = "claude-sonnet-5"
TEMPERATURE = 0.0
MAX_TOKENS = 1024
MAX_ITERATIONS = 3  # bound on the agentic loop, per credit

# Published rates for claude-sonnet-5, USD per million tokens. Used only to
# estimate the cost line in the report.
INPUT_COST_PER_MTOK = 2.00
OUTPUT_COST_PER_MTOK = 10.00

ACTIONS = ("propose_match", "widen_date_window", "request_more_candidates",
           "abstain")


@dataclass
class Proposal:
    """One model response, parsed. Not yet a match -- the gate decides that."""

    bank_txn_id: str
    proposed_payment_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    abstain: bool = False
    action: str = "propose_match"
    iterations: int = 1
    from_cache: bool = False
    error: str | None = None

    def is_actionable(self) -> bool:
        return (not self.abstain and not self.error
                and bool(self.proposed_payment_ids))


SYSTEM_PROMPT = """\
You are a reconciliation analyst for an Indian merchant's finance team.

A payment gateway collects customer payments, batches them, deducts a 2% fee \
plus 18% GST on that fee, and pushes one lump-sum credit to the merchant's \
bank account 2-3 days later. Your job is to decide which gateway payments a \
given bank credit corresponds to.

Rules you must follow:

1. The payments you propose must have net amounts that sum to the credit \
amount, within a couple of paise of rounding drift. Add them up carefully.
2. Prefer payments from a single settlement batch. A credit that spans two \
batches is unusual and needs strong evidence.
3. A payment already claimed by another credit is not available to you.
4. If no defensible answer exists, ABSTAIN. This is a normal, expected, and \
valued outcome -- not a failure. Some bank credits are vendor refunds or \
unrelated deposits that correspond to no gateway payment at all, and some \
genuinely tie between two equally plausible batches. Saying "I do not know" \
is strictly better than guessing, because a wrong match silently corrupts a \
merchant's books and nobody notices for months.
5. Never invent a payment id. Only use ids from the candidate list.

Your arithmetic will be independently re-verified against the source records \
before anything is accepted, and wrong proposals are logged. Do not guess to \
seem useful.

Respond with a single JSON object and nothing else. No prose, no markdown \
fences.

{
  "action": "propose_match" | "widen_date_window" | "request_more_candidates" \
| "abstain",
  "proposed_payment_ids": ["PAY00001", ...],
  "confidence": 0.0-1.0,
  "reasoning": "one or two sentences citing the actual amounts",
  "abstain": true | false
}

Use "widen_date_window" if the candidates look right but the settlement dates \
sit outside the window you were given. Use "request_more_candidates" if none \
of the candidates is plausible and you believe the correct payments were not \
shown to you. Both cost you an iteration, so only ask when it would change \
your answer."""


def build_prompt(cs: CandidateSet, ds: Dataset, iteration: int = 1,
                 note: str = "") -> str:
    """Render one credit and its candidates as the user message.

    Deterministic: same inputs produce the same bytes, which is what makes the
    prompt hash a usable cache key.
    """
    txn = cs.txn
    lines = [
        "Bank credit to reconcile:",
        "  transaction id : %s" % txn.txn_id,
        "  value date     : %s" % txn.value_date,
        "  amount         : %d paise (Rs %s)" % (txn.credit_paise,
                                                 format_paise(txn.credit_paise)),
        "  narration      : %s" % txn.narration,
        "",
        "Candidate payments (these are the only ids you may use):",
    ]
    by_settlement: dict[str, list] = {}
    for c in cs.candidates:
        by_settlement.setdefault(c.settlement_id, []).append(c)

    for stl in sorted(by_settlement):
        group = by_settlement[stl]
        total = cs.settlement_totals.get(stl, 0)
        lines.append("")
        lines.append("  settlement %s -- %d payments shown, unclaimed total "
                     "%d paise, settled %s (%d days before this credit)"
                     % (stl, len(group), total, group[0].payment.settled_at,
                        group[0].days_from_settlement))
        for c in sorted(group, key=lambda x: x.payment.payment_id):
            p = c.payment
            lines.append("    %s  net %8d paise  (gross %d, fee %d, gst %d)"
                         % (p.payment_id, p.net_paise, p.gross_paise,
                            p.fee_paise, p.gst_paise))

    lines.append("")
    lines.append("Difference check: the credit is %d paise. A correct answer's "
                 "net amounts must sum to that, within 2 paise."
                 % txn.credit_paise)
    if note:
        lines.append("")
        lines.append("Note for this attempt: %s" % note)
    if iteration > 1:
        lines.append("This is attempt %d of %d. If you cannot do better than "
                     "last time, abstain." % (iteration, MAX_ITERATIONS))
    return "\n".join(lines)


def prompt_hash(system: str, user: str) -> str:
    h = hashlib.sha256()
    h.update(system.encode("utf-8"))
    h.update(b"\x00")
    h.update(user.encode("utf-8"))
    h.update(("|%s|%s" % (MODEL, TEMPERATURE)).encode("utf-8"))
    return h.hexdigest()


def strip_fences(text: str) -> str:
    """Remove markdown fences the prompt asked the model not to emit.

    The instruction usually holds, but parsing must not depend on it.
    """
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    # Fall back to the outermost JSON object if prose leaked in around it.
    if not t.startswith("{"):
        start, end = t.find("{"), t.rfind("}")
        if start != -1 and end > start:
            t = t[start:end + 1]
    return t


def parse_response(text: str, bank_txn_id: str) -> Proposal:
    """Parse a model response defensively. A malformed reply is not a match."""
    try:
        data = json.loads(strip_fences(text))
    except (json.JSONDecodeError, ValueError) as exc:
        return Proposal(bank_txn_id=bank_txn_id, abstain=True,
                        error="unparseable response: %s" % exc)
    if not isinstance(data, dict):
        return Proposal(bank_txn_id=bank_txn_id, abstain=True,
                        error="response was not a JSON object")

    ids = data.get("proposed_payment_ids") or []
    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        return Proposal(bank_txn_id=bank_txn_id, abstain=True,
                        error="proposed_payment_ids was not a list of strings")

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    action = data.get("action", "propose_match")
    if action not in ACTIONS:
        action = "propose_match"

    return Proposal(
        bank_txn_id=bank_txn_id,
        proposed_payment_ids=list(dict.fromkeys(ids)),  # de-dup, keep order
        confidence=max(0.0, min(1.0, confidence)),
        reasoning=str(data.get("reasoning", ""))[:2000],
        abstain=bool(data.get("abstain", False)) or action == "abstain",
        action=action,
    )


class LLMProposer:
    """Wraps the Anthropic client with caching, logging and the agent loop."""

    def __init__(self, cache_dir="runs/cache", log_path=None,
                 k: int = DEFAULT_K, max_iterations: int = MAX_ITERATIONS,
                 client=None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.k = k
        self.max_iterations = max_iterations
        self._client = client
        self.calls = 0
        self.cache_hits = 0
        self.input_tokens = 0
        self.output_tokens = 0

    # -- client ------------------------------------------------------------

    @property
    def client(self):
        """Import and construct lazily.

        Deliberate: the module must import cleanly with no SDK installed and
        no API key present, because the offline path is a first-class
        requirement rather than a fallback.
        """
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - env dependent
                raise RuntimeError(
                    "the anthropic package is not installed; run with "
                    "--no-llm for the deterministic pipeline") from exc
            self._client = anthropic.Anthropic()
        return self._client

    # -- cache -------------------------------------------------------------

    def _cache_path(self, digest: str) -> Path:
        return self.cache_dir / ("%s.json" % digest)

    def _cache_get(self, digest: str):
        path = self._cache_path(digest)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None  # a corrupt cache entry is a miss, not a crash

    def _cache_put(self, digest: str, payload: dict) -> None:
        self._cache_path(digest).write_text(
            json.dumps(payload, indent=2), encoding="utf-8")

    # -- logging -----------------------------------------------------------

    def _log(self, record: dict) -> None:
        if not self.log_path:
            return
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    # -- the call ----------------------------------------------------------

    def _call(self, system: str, user: str, txn_id: str,
              candidates: list[str], iteration: int) -> tuple[str, dict]:
        """One model call, cache-first. Returns (raw_text, metadata)."""
        digest = prompt_hash(system, user)
        cached = self._cache_get(digest)
        if cached is not None:
            self.cache_hits += 1
            meta = {"prompt_hash": digest, "cached": True, "latency_ms": 0,
                    "input_tokens": cached.get("input_tokens", 0),
                    "output_tokens": cached.get("output_tokens", 0)}
            self._log({"bank_txn_id": txn_id, "iteration": iteration,
                       "candidates": candidates, "raw_response": cached["text"],
                       **meta})
            return cached["text"], meta

        start = time.perf_counter()
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        text = "".join(block.text for block in response.content
                       if getattr(block, "type", "") == "text")
        usage = getattr(response, "usage", None)
        in_tok = getattr(usage, "input_tokens", 0) if usage else 0
        out_tok = getattr(usage, "output_tokens", 0) if usage else 0

        self.calls += 1
        self.input_tokens += in_tok
        self.output_tokens += out_tok

        self._cache_put(digest, {"text": text, "input_tokens": in_tok,
                                 "output_tokens": out_tok, "model": MODEL})
        meta = {"prompt_hash": digest, "cached": False,
                "latency_ms": latency_ms, "input_tokens": in_tok,
                "output_tokens": out_tok,
                "estimated_cost_usd": self._cost(in_tok, out_tok)}
        self._log({"bank_txn_id": txn_id, "iteration": iteration,
                   "candidates": candidates, "raw_response": text, **meta})
        return text, meta

    @staticmethod
    def _cost(input_tokens: int, output_tokens: int) -> float:
        return round(input_tokens / 1e6 * INPUT_COST_PER_MTOK
                     + output_tokens / 1e6 * OUTPUT_COST_PER_MTOK, 6)

    # -- the agentic loop --------------------------------------------------

    def propose(self, ds: Dataset, txn: BankTxn, claimed: set[str]) -> Proposal:
        """Propose a match for one credit, with a bounded agent loop.

        The model chooses its next action rather than only answering once: it
        may ask for a wider date window or more candidates before committing,
        or abstain outright. Bounded at ``max_iterations`` so a confused model
        cannot loop indefinitely, and every proposal it does make still passes
        through the verification gate untouched.
        """
        from .retrieve import RETRIEVAL_WINDOW_DAYS

        window = RETRIEVAL_WINDOW_DAYS
        k = self.k
        note = ""
        last: Proposal | None = None

        for iteration in range(1, self.max_iterations + 1):
            cs = retrieve(ds, txn, claimed, k=k, window_days=window)
            if not cs.candidates:
                return Proposal(bank_txn_id=txn.txn_id, abstain=True,
                                reasoning="no candidate payments available "
                                          "within the retrieval window",
                                action="abstain", iterations=iteration)

            user = build_prompt(cs, ds, iteration=iteration, note=note)
            try:
                text, meta = self._call(SYSTEM_PROMPT, user, txn.txn_id,
                                        sorted(cs.payment_ids), iteration)
            except Exception as exc:  # noqa: BLE001 - any failure is an abstain
                return Proposal(bank_txn_id=txn.txn_id, abstain=True,
                                error="llm call failed: %s" % exc,
                                iterations=iteration)

            proposal = parse_response(text, txn.txn_id)
            proposal.iterations = iteration
            proposal.from_cache = meta.get("cached", False)
            last = proposal

            if proposal.action == "widen_date_window" and iteration < self.max_iterations:
                window += 3
                note = ("date window widened to %d days at your request"
                        % window)
                continue
            if proposal.action == "request_more_candidates" and iteration < self.max_iterations:
                k = min(k * 2, 60)
                note = "candidate list widened to %d payments at your request" % k
                continue

            # propose_match or abstain: done either way.
            return proposal

        return last or Proposal(bank_txn_id=txn.txn_id, abstain=True,
                                reasoning="exhausted iterations without a "
                                          "defensible answer",
                                action="abstain",
                                iterations=self.max_iterations)

    def stats(self) -> dict:
        return {
            "llm_calls": self.calls,
            "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": self._cost(self.input_tokens,
                                             self.output_tokens),
            "model": MODEL,
        }

"""Command-line interface.

    generate   synthetic data + ground truth
    validate   ground-truth self-consistency checks
    run        the reconciliation pipeline, with or without the LLM layer
    sweep      the same pipeline at several difficulty mixes
    selftest   prove the verification gate rejects corrupted matches
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import generate as gen
from .validate import validate


def cmd_generate(args) -> int:
    counts = gen.generate(args.seed, args.hard_ratio, args.out)
    out = Path(args.out)
    print("Generated into %s/" % out)
    print("  orders.csv       %4d rows" % counts["orders"])
    print("  settlements.csv  %4d rows" % counts["payments"])
    print("  bank.csv         %4d rows" % counts["bank_rows"])
    print("  truth.json       %4d settlement batches" % counts["batches"])
    print("  seed=%d hard_ratio=%.2f" % (args.seed, args.hard_ratio))
    return 0


def cmd_validate(args) -> int:
    failures = validate(args.data_dir)
    if not failures:
        print("VALIDATION PASSED - ground truth is self-consistent (%s)"
              % args.data_dir)
        return 0
    print("VALIDATION FAILED - %d problem(s):" % len(failures), file=sys.stderr)
    for f in failures:
        print("  - %s" % f, file=sys.stderr)
    return 1


def cmd_run(args) -> int:
    """Run the reconciliation pipeline and report metrics.

    The deterministic baseline is measured and persisted before the LLM layer
    is called at all, so the ablation compares like with like: no later stage
    can quietly re-run the deterministic stages under different conditions.
    """
    import json
    import time
    from datetime import datetime

    from .evaluate import evaluate, format_metrics_table, per_case_breakdown
    from .ingest import load_dataset, load_truth
    from .match_deterministic import run_deterministic
    from .models import Match

    started = time.perf_counter()
    ds = load_dataset(args.data_dir)
    truth = load_truth(args.data_dir)

    # Order-side: an order with no payment record can never appear as a bank
    # match, so it is detected here rather than by any matching stage.
    paid_order_ids = {p.order_id for p in ds.payments}
    detected_unpaid = {o.order_id for o in ds.orders
                       if o.order_id not in paid_order_ids}

    print("Loaded %d orders, %d payments, %d bank rows (%d credits, %d debits)"
          % (len(ds.orders), len(ds.payments), len(ds.bank),
             len(ds.credits), len(ds.debits)))
    print()

    # Ablation: each stage cut is a fresh run, so the rows are independent
    # measurements rather than a running total.
    ablation = []
    for n, label in [(1, "Stage 1 only (reference)"),
                     (2, "Stages 1-2 (+ group sum)"),
                     (3, "Stages 1-3 (deterministic baseline)")]:
        r = run_deterministic(ds, stages=n)
        met = evaluate(r.matches, truth, ds, detected_unpaid)
        ablation.append((label, met, r))

    label, metrics, result = ablation[-1]
    elapsed = time.perf_counter() - started

    print("Ablation")
    print("=" * 72)
    print("  %-38s %9s %9s %6s" % ("Configuration", "Auto-match", "Precision",
                                   "Forced"))
    for lbl, met, _ in ablation:
        print("  %-38s %8.1f%% %8.1f%% %6d"
              % (lbl, 100 * met.auto_match_rate, 100 * met.precision,
                 met.forced_match_errors))
    print()
    print(format_metrics_table(metrics, "Baseline (deterministic, stages 1-3)"))
    print()

    print("Per-case breakdown")
    print("=" * 72)
    for case, row in per_case_breakdown(result.matches, truth).items():
        print("  %-20s %2d/%2d correct  %5.0f%%%s"
              % (case, row["correct"], row["total"], 100 * row["rate"],
                 "   <-- %d WRONG" % row["wrong"] if row["wrong"] else ""))
    print()

    print("Stage contributions")
    print("=" * 72)
    for stage, count in result.stage_counts.items():
        print("  %-20s %3d matches" % (stage, count))
    print("  %-20s %3d credits" % ("unresolved", len(result.unresolved)))
    print()

    print("Throughput")
    print("=" * 72)
    records = len(ds.orders) + len(ds.payments) + len(ds.bank)
    print("  %d records in %.3fs  (%.0f records/sec)"
          % (records, elapsed, records / elapsed if elapsed else 0))
    print()

    run_dir = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- LLM proposal layer -----------------------------------------------
    # Nothing the model proposes is a match until verify.py passes it. The
    # metrics printed above are the deterministic baseline and are already
    # persisted, so no proposal can retroactively change them.
    llm_stats = None
    gate_summary = None
    proposals = []
    if not args.no_llm:
        from .match_llm import LLMProposer

        residual = [t for t in ds.credits
                    if t.txn_id not in result.matched_txn_ids]
        from .providers import get_provider

        print("LLM proposal layer")
        print("=" * 72)

        # Preflight. An unreachable model would abstain on everything, which
        # looks like a cautious model rather than a missing key -- so refuse
        # to run rather than emit a number that reads as a measurement.
        provider = get_provider(args.provider, model=args.model)
        usable, why = provider.available()
        if not usable:
            print("  cannot reach the %s provider: %s" % (args.provider, why),
                  file=sys.stderr)
            print("  Every proposal would abstain, which is NOT a measurement",
                  file=sys.stderr)
            print("  of abstention. Fix the above, or use --no-llm.",
                  file=sys.stderr)
            return 3

        print("  provider: %s   model: %s" % (args.provider, provider.model))
        proposer = LLMProposer(cache_dir=Path(args.out) / "cache",
                               log_path=run_dir / "llm_calls.jsonl",
                               provider=args.provider, model=args.model)

        print("  %d residual credits after the deterministic stages"
              % len(residual))
        for txn in residual:
            p = proposer.propose(ds, txn, result.claimed_payment_ids)
            proposals.append(p)
            print("  %-9s %-22s conf=%.2f  %s"
                  % (txn.txn_id,
                     "ABSTAIN" if p.abstain else "propose %d payments"
                     % len(p.proposed_payment_ids),
                     p.confidence, p.reasoning[:60]))
        abstained = sum(1 for p in proposals if p.abstain)
        failed = sum(1 for p in proposals if p.error)
        genuine_abstentions = abstained - failed
        llm_stats = proposer.stats()

        if failed:
            print()
            print("  %d of %d calls FAILED. A failed call is not an abstention."
                  % (failed, len(proposals)), file=sys.stderr)
            for p in proposals:
                if p.error:
                    print("    %s: %s" % (p.bank_txn_id, p.error[:150]),
                          file=sys.stderr)
        if failed == len(proposals) and proposals:
            print("  Every call failed, so there is nothing to measure and no",
                  file=sys.stderr)
            print("  report worth writing. Fix the provider error above, or",
                  file=sys.stderr)
            print("  use --no-llm for a clean deterministic run.",
                  file=sys.stderr)
            return 4

        print("  abstention rate: %d/%d (%.0f%%)%s"
              % (genuine_abstentions, len(proposals),
                 100 * genuine_abstentions / len(proposals) if proposals else 0,
                 "  [%d failed calls excluded]" % failed if failed else ""))
        print("  %s" % llm_stats)
        print()

        # --- Verification gate --------------------------------------------
        # Nothing the model proposed is a match until this passes it. The
        # gate recomputes every amount from source and knows nothing about
        # where the proposal came from.
        from .verify import verify_all

        actionable = [p for p in proposals if p.is_actionable()]
        gate = verify_all(actionable, ds, result.claimed_payment_ids,
                          confidence_threshold=args.confidence_threshold)
        gate_summary = gate.summary()

        print("Verification gate")
        print("=" * 72)
        print("  proposals made      %3d" % len(proposals))
        print("  ...abstained        %3d (never reached the gate)"
              % (len(proposals) - len(actionable)))
        print("  ...verified         %3d" % gate_summary["proposals_verified"])
        print("  accepted            %3d" % gate_summary["accepted"])
        print("  rejected            %3d" % gate_summary["rejected"])
        for rule, count in gate_summary["rejections_by_rule"].items():
            print("      %-22s %2d" % (rule, count))
        for v in gate.rejected:
            print("  REJECTED %s: %s" % (v.bank_txn_id, v.reason[:70]))
        print()

        # The rejection log is a deliverable: it is the evidence that the
        # safety layer does real work rather than passing everything through.
        with open(run_dir / "rejections.jsonl", "w", encoding="utf-8") as fh:
            for v in gate.rejected:
                fh.write(json.dumps(v.as_log_record(), sort_keys=True) + "\n")

        gate_summary["proposals_made"] = len(proposals)
        gate_summary["abstained"] = len(proposals) - len(actionable)
        gate_summary["rejections"] = [
            {"bank_txn_id": v.bank_txn_id, "failed_rule": str(v.failed_rule),
             "reason": v.reason} for v in gate.rejected]
        # Rupee value of what the gate refused. Finance panels respond to
        # money, not F1 scores -- and these are synthetic funds, said plainly.
        gate_summary["cost_of_being_wrong_paise"] = sum(
            ds.bank_by_id[v.bank_txn_id].credit_paise for v in gate.rejected
            if v.bank_txn_id in ds.bank_by_id)

        # Accepted proposals become real matches and are re-scored.
        for v in gate.accepted:
            result.matches.append(Match(
                bank_txn_id=v.bank_txn_id, payment_ids=v.payment_ids,
                stage="llm_verified",
                confidence=v.detail.get("confidence", 0.0),
                reasoning=v.reason))
            result.unresolved.pop(v.bank_txn_id, None)

        final_metrics = evaluate(result.matches, truth, ds, detected_unpaid)
        ablation.append(("Stages 1-5 (+ LLM, verified)", final_metrics, result))

        print(format_metrics_table(final_metrics,
                                   "Final (deterministic + verified LLM)"))
        print()
        print("  delta vs baseline: auto-match %+.1f pp, precision %+.1f pp"
              % (100 * (final_metrics.auto_match_rate - metrics.auto_match_rate),
                 100 * (final_metrics.precision - metrics.precision)))
        print()

    # Persist the baseline. SPEC is explicit that these numbers must not be
    # lost or overwritten when the LLM layer lands.
    baseline = {
        "generated_from": str(args.data_dir),
        "seed": truth.seed,
        "hard_ratio": truth.hard_ratio,
        "llm_enabled": not args.no_llm,
        "elapsed_seconds": round(elapsed, 4),
        "records": records,
        "ablation": [{"configuration": lbl, **met.as_dict()}
                     for lbl, met, _ in ablation],
        "baseline": metrics.as_dict(),
        "per_case": per_case_breakdown(result.matches, truth),
        "stage_counts": result.stage_counts,
        "unresolved": result.unresolved,
        "ambiguous": result.ambiguous,
        "llm": llm_stats,
        "gate": gate_summary,
        "proposals": [
            {"bank_txn_id": pr.bank_txn_id,
             "proposed_payment_ids": pr.proposed_payment_ids,
             "confidence": pr.confidence, "abstain": pr.abstain,
             "action": pr.action, "iterations": pr.iterations,
             "reasoning": pr.reasoning, "error": pr.error}
            for pr in proposals
        ],
    }
    (run_dir / "baseline.json").write_text(
        json.dumps(baseline, indent=2), encoding="utf-8")

    # --- Exception classification and report ------------------------------
    from .classify import classification_accuracy, classify_all
    from .report import build_report, cash_position, git_commit

    final_metrics = ablation[-1][1]
    classified = classify_all(ds, result.unresolved, result.ambiguous,
                              detected_unpaid)
    accuracy = classification_accuracy(classified, truth)

    print("Exception list")
    print("=" * 72)
    for c in classified:
        print("  %-10s %-20s %s" % (c.exception.record_id, c.category,
                                    c.reason[:56]))
    print("  classification accuracy vs truth: %d/%d (%.0f%%)"
          % (accuracy["correct"], accuracy["scored"],
             100 * accuracy["accuracy"]))
    print()

    report_ctx = {
        "git_commit": git_commit(),
        "seed": truth.seed,
        "hard_ratio": truth.hard_ratio,
        "counts": {"orders": len(ds.orders), "payments": len(ds.payments),
                   "bank": len(ds.bank), "credits": len(ds.credits),
                   "debits": len(ds.debits),
                   "settlements": len(ds.payments_by_settlement)},
        "elapsed": elapsed,
        "records": records,
        "throughput": records / elapsed if elapsed else 0,
        "metrics": final_metrics,
        "baseline_metrics": metrics,
        "ablation": [(lbl, met) for lbl, met, _ in ablation],
        "exceptions": classified,
        "classification_accuracy": accuracy,
        "cash": cash_position(ds, result.matches),
        "gate": gate_summary,
        "llm_stats": llm_stats,
        "llm_ran": bool(llm_stats),
        "llm_description": ("%s / %s" % (llm_stats["provider"],
                                         llm_stats["model"]))
                           if llm_stats else "not used (--no-llm)",
        "sweep": _load_sweep(args),
    }
    (run_dir / "report.md").write_text(build_report(report_ctx),
                                       encoding="utf-8")
    print("Artifacts")
    print("=" * 72)
    print("  %s" % (run_dir / "baseline.json"))
    print("  %s" % (run_dir / "report.md"))
    if gate_summary:
        print("  %s" % (run_dir / "rejections.jsonl"))
        print("  %s" % (run_dir / "llm_calls.jsonl"))
    return 0


def _load_sweep(args):
    """Reuse a sweep.json sitting beside the run, if one was produced."""
    import json
    path = Path(args.out) / "sweep.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def cmd_selftest(args) -> int:
    """Prove the verification gate rejects deliberately corrupted matches.

    The rejection log shows the gate rejected something; this shows it rejects
    what it *should*, on demand, without anyone having to trust that the
    logged rejections were representative.
    """
    import json

    from .ingest import load_dataset
    from .match_deterministic import run_deterministic
    from .selftest import format_selftest, run_selftest

    ds = load_dataset(args.data_dir)
    result = run_deterministic(ds, stages=3)
    results, all_caught = run_selftest(ds, result.matches,
                                       result.claimed_payment_ids)
    if not results:
        print("no multi-payment matches available to corrupt", file=sys.stderr)
        return 1

    print(format_selftest(results))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("\nwritten to %s" % out)

    if not all_caught:
        print("\nSELF-TEST FAILED: the gate missed at least one corruption",
              file=sys.stderr)
        return 1
    print("\nSELF-TEST PASSED: every injected corruption was rejected")
    return 0


def cmd_sweep(args) -> int:
    """Measure the pipeline at several difficulty mixes.

    A single match rate at one difficulty is easy to over-read. Three points
    show whether the number reflects the system or the dataset.
    """
    import json
    import tempfile

    from .evaluate import evaluate
    from .generate import generate
    from .ingest import load_dataset, load_truth
    from .match_deterministic import run_deterministic

    ratios = [float(r) for r in args.ratios.split(",")]
    rows = []

    print("Difficulty sweep")
    print("=" * 72)
    print("  %-11s %-15s %11s %10s %7s"
          % ("hard_ratio", "clean batches", "auto-match", "precision", "forced"))

    for ratio in ratios:
        with tempfile.TemporaryDirectory() as tmp:
            generate(args.seed, ratio, tmp)
            ds = load_dataset(tmp)
            truth = load_truth(tmp)
            paid = {p.order_id for p in ds.payments}
            unpaid = {o.order_id for o in ds.orders if o.order_id not in paid}
            result = run_deterministic(ds, stages=3)
            met = evaluate(result.matches, truth, ds, unpaid)

            cases = {}
            for tm in truth.matches:
                cases[str(tm.case)] = cases.get(str(tm.case), 0) + 1
            clean = cases.get("clean_batch", 0)
            total = sum(cases.values())

            rows.append({
                "hard_ratio": ratio, "batches": total, "clean_batches": clean,
                "auto_match_rate": met.auto_match_rate,
                "precision": met.precision, "recall": met.recall,
                "forced_match_errors": met.forced_match_errors,
                "true_positives": met.true_positives,
                "matchable_credits": met.matchable_credits,
            })
            print("  %-11.1f %-15s %10.1f%% %9.1f%% %7d"
                  % (ratio, "%d / %d" % (clean, total),
                     100 * met.auto_match_rate, 100 * met.precision,
                     met.forced_match_errors))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep.json").write_text(json.dumps(rows, indent=2),
                                    encoding="utf-8")
    print()
    print("  written to %s" % (out / "sweep.json"))
    print("  the next `run` will include this table in its report")
    return 0



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m recon.cli",
        description="Multi-source reconciliation agent")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate synthetic data + ground truth")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--hard-ratio", type=float, default=0.4,
                   dest="hard_ratio", help="target share of non-clean batches")
    g.add_argument("--out", default="data/")
    g.set_defaults(func=cmd_generate)

    v = sub.add_parser("validate", help="check ground-truth self-consistency")
    v.add_argument("data_dir", nargs="?", default="data/")
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("run", help="run the reconciliation pipeline")
    r.add_argument("data_dir", nargs="?", default="data/")
    r.add_argument("--no-llm", action="store_true")
    r.add_argument("--confidence-threshold", type=float, default=0.7)
    r.add_argument("--provider", default="anthropic",
                   choices=["anthropic", "groq", "gemini"],
                   help="LLM provider (default: anthropic, per SPEC)")
    r.add_argument("--model", default=None,
                   help="override the provider's default model")
    r.add_argument("--out", default="runs/")
    r.set_defaults(func=cmd_run)

    st = sub.add_parser("selftest",
                        help="prove the verification gate rejects corrupted matches")
    st.add_argument("data_dir", nargs="?", default="data/")
    st.add_argument("--out", default=None, help="write results as JSON")
    st.set_defaults(func=cmd_selftest)

    s = sub.add_parser("sweep", help="run at several difficulty ratios")
    s.add_argument("data_dir", nargs="?", default="data/")
    s.add_argument("--ratios", default="0.2,0.4,0.6")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default="runs/")
    s.set_defaults(func=cmd_sweep)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

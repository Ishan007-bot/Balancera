"""Command-line interface.

Phase 1 implements ``generate`` and ``validate``. ``run`` and ``sweep`` are
declared but not yet implemented -- they land in Phases 2 and 5. They exit
with a clear message rather than a stack trace so the surface is honest about
what exists today.
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

    Phase 2: deterministic stages 1-3 only. The baseline is persisted before
    any LLM layer exists, so the later comparison cannot be contaminated by
    re-running the deterministic stages under different conditions.
    """
    import json
    import time
    from datetime import datetime

    from .evaluate import evaluate, format_metrics_table, per_case_breakdown
    from .ingest import load_dataset, load_truth
    from .match_deterministic import run_deterministic

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

    # Persist the baseline. SPEC is explicit that these numbers must not be
    # lost or overwritten when the LLM layer lands.
    run_dir = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
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
    }
    (run_dir / "baseline.json").write_text(
        json.dumps(baseline, indent=2), encoding="utf-8")
    print("Baseline written to %s" % (run_dir / "baseline.json"))

    if not args.no_llm:
        print()
        print("note: LLM layer lands in Phase 3; this run is deterministic only.")
    return 0


def cmd_not_yet(phase: str):
    def _run(args) -> int:
        print("not implemented yet - lands in %s" % phase, file=sys.stderr)
        return 2
    return _run


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
    r.add_argument("--out", default="runs/")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("sweep", help="run at several difficulty ratios")
    s.add_argument("data_dir", nargs="?", default="data/")
    s.add_argument("--ratios", default="0.2,0.4,0.6")
    s.set_defaults(func=cmd_not_yet("Phase 5"))

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

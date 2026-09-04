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
    r.set_defaults(func=cmd_not_yet("Phase 2"))

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

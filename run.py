"""Cross-platform task runner.

`make` is not installed on every machine (notably Windows), and SPEC section
13 makes `make demo` working from a clean clone a hard requirement. The
Makefile therefore delegates every target here, so both of these work and do
exactly the same thing:

    make demo
    python run.py demo

No dependencies, no API key, nothing to install.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PY = sys.executable


def sh(*args: str) -> int:
    """Run a subcommand, echoing it first so the demo is legible on camera."""
    print("$ " + " ".join(args), flush=True)
    return subprocess.call(list(args), cwd=ROOT)


def step(title: str) -> None:
    print("\n" + "=" * 68)
    print("  " + title)
    print("=" * 68, flush=True)


def demo(with_llm: bool = False) -> int:
    """Generate, validate, sweep, self-test, reconcile -- end to end.

    The sweep and self-test run *before* the reconciliation, because the
    report reads sweep.json when it renders section 4. Running them after
    would leave a reviewer following the quickstart with an empty table.
    """
    step("1/5  Generate synthetic data and ground truth")
    rc = sh(PY, "-m", "recon.cli", "generate", "--seed", "42", "--out", "data/")
    if rc:
        return rc

    step("2/5  Validate ground-truth self-consistency")
    rc = sh(PY, "-m", "recon.cli", "validate", "data/")
    if rc:
        return rc

    step("3/5  Difficulty sweep (feeds section 4 of the report)")
    rc = sh(PY, "-m", "recon.cli", "sweep", "data/", "--ratios", "0.2,0.4,0.6")
    if rc:
        return rc

    step("4/5  Adversarial self-test of the verification gate")
    rc = sh(PY, "-m", "recon.cli", "selftest", "data/")
    if rc:
        return rc

    step("5/5  Reconcile" + (" (LLM enabled)" if with_llm else " (offline)"))
    cmd = [PY, "-m", "recon.cli", "run", "data/"]
    if not with_llm:
        cmd.append("--no-llm")
    return sh(*cmd)


def test() -> int:
    return sh(PY, "-m", "pytest", "tests/", "-q")


def clean() -> int:
    # data/sample/ is a committed fixture, not generated output -- a reviewer
    # should be able to inspect real data without running anything. Remove
    # everything else under data/ but leave it alone.
    data = ROOT / "data"
    if data.exists():
        for child in data.iterdir():
            if child.name == "sample":
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink()
        print("removed data/ (kept data/sample/)")
    if (ROOT / "runs").exists():
        shutil.rmtree(ROOT / "runs", ignore_errors=True)
        print("removed runs/")
    for cache in ROOT.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    for cache in [ROOT / ".pytest_cache"]:
        shutil.rmtree(cache, ignore_errors=True)
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / ".gitkeep").touch()
    print("cleaned")
    return 0


TASKS = {
    "demo": lambda: demo(with_llm=False),
    "demo-llm": lambda: demo(with_llm=True),
    "test": test,
    "clean": clean,
}


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in TASKS:
        print("usage: python run.py {%s}" % "|".join(TASKS), file=sys.stderr)
        return 2
    return TASKS[argv[1]]()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

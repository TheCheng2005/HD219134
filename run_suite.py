"""Run the whole HD 219134 suite unattended.

    python run_suite.py                 check the setup, then run every unfinished or failed run
    python run_suite.py --list          show runs and their status
    python run_suite.py --only paper_M4,nested_M4+f
    python run_suite.py --group paper   (paper, nested, kprior, prot; comma-separated)
    python run_suite.py --jobs 2        runs at once (default: CPU cores // 4)
    python run_suite.py --skip-failed   leave runs that failed before alone
    python run_suite.py --quick         tiny end-to-end test of the pipeline (results_quick/)
    python run_suite.py --summary       only rebuild results/summary/

Finished runs are skipped, so after a crash, reboot or Ctrl-C just start it again.
Interrupted runs restart from the beginning (a run cannot resume mid-sampling).
Run only one copy at a time: two copies would write to the same result folders.
"""
import argparse
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from hd219134.run_one import ROOT, results_dir

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def keep_awake():
    """Stop Windows from sleeping while the suite runs (the display may still turn off)."""
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--group", default="")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--skip-failed", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args()

    from hd219134.runs import QUICK_SAMPLING, RUNS, SAMPLING
    from hd219134.summarize import status, summarize

    base = results_dir(args.quick)
    base.mkdir(exist_ok=True)
    runs = RUNS
    if args.only:
        wanted = args.only.split(",")
        unknown = set(wanted) - {r.id for r in RUNS}
        if unknown:
            sys.exit(f"unknown run ids: {sorted(unknown)}  (see --list)")
        runs = [r for r in RUNS if r.id in wanted]
    if args.group:
        runs = [r for r in runs if r.group in args.group.split(",")]

    if args.list:
        for r in runs:
            print(f"{status(base, r):8s} {r.id:18s} {r.group:7s} {r.note}")
        return
    if args.summary:
        print(summarize(base))
        return

    if not args.skip_check:
        print("Checking setup (tests/check_setup.py) ...", flush=True)
        if subprocess.call([sys.executable, str(ROOT / "tests" / "check_setup.py")], cwd=ROOT) != 0:
            sys.exit("Setup check failed; fix the problem above (or use --skip-check if you know better).")

    # FAILED runs are retried by default: Ctrl-C also ends runs with a non-zero exit code
    todo = [r for r in runs if status(base, r) in ("pending", "running") or
            (status(base, r) == "FAILED" and not args.skip_failed)]
    chains = (QUICK_SAMPLING if args.quick else SAMPLING)["chains"]
    jobs = args.jobs or max(1, (os.cpu_count() or chains) // chains)
    print(f"{len(todo)} runs to do, {jobs} at a time ({chains} chains each, {os.cpu_count()} CPU cores). "
          f"Results in {base}", flush=True)
    keep_awake()
    lock = threading.Lock()

    def work(run):
        d = base / run.id
        d.mkdir(parents=True, exist_ok=True)
        for f in ("failed.txt", "done.json"):
            (d / f).unlink(missing_ok=True)
        (d / "running.txt").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
        cmd = [sys.executable, "-m", "hd219134.run_one", run.id] + (["--quick"] if args.quick else [])
        with lock:
            print(f"[{time.strftime('%H:%M:%S')}] start  {run.id}  (log: {d / 'log.txt'})", flush=True)
        # UTF-8 so non-ASCII output can't crash a run on Windows (log files default to cp1252);
        # unbuffered so log.txt is current
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        with open(d / "log.txt", "ab") as log:
            code = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        (d / "running.txt").unlink(missing_ok=True)
        if code != 0:
            (d / "failed.txt").write_text(f"exit code {code}; see log.txt\n", encoding="utf-8")
        with lock:
            print(f"[{time.strftime('%H:%M:%S')}] {'done ' if code == 0 else 'FAILED'} {run.id}", flush=True)
            try:
                summarize(base)
            except Exception:
                traceback.print_exc()

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(work, todo))
    try:
        print(f"\nAll done. Summary: {summarize(base)}")
    except Exception:  # e.g. a summary file left open in Excel, which locks it on Windows
        traceback.print_exc()
        print("\nAll runs finished, but the summary could not be written; close any open summary "
              "files and run: python run_suite.py --summary")


if __name__ == "__main__":
    main()

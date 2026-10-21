"""Check that this machine can run the suite and computes the same model as the reference machine.

    python tests/check_setup.py                    run all checks (about 2-5 minutes)
    python tests/check_setup.py --write-reference  regenerate tests/reference_logp.json (reference machine only)
"""
import json
import pathlib
import sys
import time
import warnings

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REFERENCE = ROOT / "tests" / "reference_logp.json"
EXPECTED = {"pymc": "5.25.1", "pytensor": "2.31.7", "exoplanet": "0.6.0", "celerite2": "0.3.2"}

warnings.filterwarnings("ignore")
failures = []


def check(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg, flush=True)
    if not ok:
        failures.append(msg)


def main(write_reference=False):
    import numpy as np
    import pymc as pm
    import pytensor
    import exoplanet
    import celerite2

    print("1. Packages")
    for mod in (pm, pytensor, exoplanet, celerite2):
        name = mod.__name__
        check(mod.__version__ == EXPECTED[name], f"{name} {mod.__version__} (expected {EXPECTED[name]})")
    check(bool(pytensor.config.cxx), f"C compiler for pytensor: '{pytensor.config.cxx}' "
          "(empty means sampling would be 10-100x slower; see README, Windows setup)")

    print("2. Data")
    from hd219134.data import check_files, load_data
    try:
        check_files()
        check(True, "all TESS and RV files present")
    except FileNotFoundError as e:
        check(False, str(e))
        return
    data = load_data(verbose=False)
    check(len(data.x) == 79862 and len(data.x_rv) == 2024, f"{len(data.x)} TESS points, {len(data.x_rv)} RVs "
          "(expected 79862 and 2024)")

    print("3. Every run's model at its starting point")
    from hd219134.metrics import make_point_builder
    from hd219134.model import build_model, inject_rv_signal
    from hd219134.runs import RUNS, SAMPLING, chain_initvals

    ref = json.loads(REFERENCE.read_text(encoding="utf-8")) if REFERENCE.exists() and not write_reference else {}
    new_ref = {}
    timing = None
    for run in RUNS:
        d = inject_rv_signal(data, *run.injection) if run.injection else data
        model = build_model(d, run.planets, catalog=run.catalog, circular=run.circular, prot_prior=run.prot_prior)
        point = make_point_builder(model)(chain_initvals(run, 1)[0])
        f = model.compile_fn([model.logp(), *pytensor.grad(model.logp(), model.value_vars)],
                             inputs=model.value_vars, on_unused_input="ignore")
        lp, *g = f(point)
        finite = np.isfinite(lp) and all(np.all(np.isfinite(gi)) for gi in g)
        new_ref[run.id] = float(lp)
        if run.id in ref:
            check(finite and abs(lp - ref[run.id]) < 1e-2,
                  f"{run.id:18s} logp {lp:.6f} (reference {ref[run.id]:.6f})")
        else:
            check(finite, f"{run.id:18s} logp {lp:.6f} finite, gradient finite (no reference value)")
        if run.id == "main_M4":
            f(point)
            t = time.time()
            for _ in range(10):
                f(point)
            timing = (time.time() - t) / 10

    print("4. Injection")
    x = inject_rv_signal(data, 27.0, 2.3, 2457000.0).y_rv - data.y_rv
    check(abs((x.max() - x.min()) / 2 - 2.3) < 0.02, f"injected semi-amplitude {(x.max() - x.min()) / 2:.3f} m/s "
          "(expected 2.30)")

    if write_reference:
        REFERENCE.write_text(json.dumps(new_ref, indent=1), encoding="utf-8")
        print(f"wrote {REFERENCE}")

    n_iter = SAMPLING["tune"] + SAMPLING["draws"]
    print(f"\nOne log-probability + gradient evaluation: {timing * 1e3:.1f} ms.")
    print(f"Rough time per run ({n_iter} iterations per chain): about {timing * 63 * n_iter / 3600:.0f} h when the "
          f"sampler behaves like the converged 4-planet run (63 steps/draw), up to about "
          f"{timing * 1023 * n_iter / 3600:.0f} h when it hits the tree-depth limit (1023 steps/draw).")


if __name__ == "__main__":
    main(write_reference="--write-reference" in sys.argv)
    if failures:
        print(f"\n{len(failures)} check(s) failed.")
        sys.exit(1)
    print("\nAll checks passed.")

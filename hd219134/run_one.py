"""Sample and analyze one run:  python -m hd219134.run_one <run_id> [--quick]

Writes results/<run_id>/{trace.nc, metrics.json, summary.csv, done.json}.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import dataclasses
import json
import pathlib
import platform
import sys
import time
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent


def results_dir(quick):
    return ROOT / ("results_quick" if quick else "results")


def progress_callback(total, every, log):
    """Log one line per chain every `every` iterations. Replaces PyMC's live progress bar,
    which floods log files and uses characters a Windows log file cannot encode."""
    t0 = time.time()
    divergences = {}

    def callback(trace, draw):
        stats = draw.stats[0] if isinstance(draw.stats, (list, tuple)) else draw.stats
        if not draw.tuning and stats.get("diverging", False):
            divergences[draw.chain] = divergences.get(draw.chain, 0) + 1
        n = draw.draw_idx + 1
        if n % every == 0 or draw.is_last:
            log(f"chain {draw.chain}: iteration {n}/{total} ({'tuning' if draw.tuning else 'sampling'}), "
                f"{divergences.get(draw.chain, 0)} divergences so far, {(time.time() - t0) / 3600:.2f} h elapsed")

    return callback


def main(run_id, quick=False):
    import arviz as az
    import pymc as pm
    import pytensor

    from .data import load_data
    from .metrics import analyze
    from .model import build_model, inject_rv_signal
    from .runs import QUICK_SAMPLING, RUNS_BY_ID, SAMPLING, chain_initvals

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="arviz")  # R-hat of constant parameters
    run = RUNS_BY_ID[run_id]
    cfg = QUICK_SAMPLING if quick else SAMPLING
    out = results_dir(quick) / run_id
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log = lambda msg: print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

    log(f"run {run.id}: {run.note}")
    log(f"planets {run.planets}, K-prior planets {run.k_planets}, P_rot prior {run.prot_prior}, "
        f"injection {run.injection}, sampling {cfg}, seed {run.seed}")

    data = load_data(verbose=False)
    if run.injection:
        P, K, t0 = run.injection
        data = inject_rv_signal(data, P, K, t0)
    model = build_model(data, run.planets, catalog=run.catalog, k_planets=run.k_planets,
                        prot_prior=run.prot_prior)

    with model:
        trace = pm.sample(
            tune=cfg["tune"], draws=cfg["draws"], chains=cfg["chains"], cores=cfg["chains"],
            init="adapt_full", initvals=chain_initvals(run, cfg["chains"]),
            target_accept=cfg["target_accept"], random_seed=run.seed,
            return_inferencedata=True, compute_convergence_checks=False, progressbar=False,
            callback=progress_callback(cfg["tune"] + cfg["draws"], 5 if quick else 100, log),
        )
    trace.to_netcdf(out / "trace.nc")
    t_sampled = time.time()
    log(f"sampling done in {(t_sampled - t_start) / 3600:.2f} h; analyzing")

    metrics, summary = analyze(model, trace, data.n_obs, list(run.planets), k_planets=run.k_planets, progress=log)
    summary_full = az.summary(trace, var_names=["P_rot", "period", "K", "m_pl", "ecc", "rv_offsets",
                                                "rv_jitters", "C_activity", "log_sigma_rv_gp", "m_star", "r_star"])
    summary_full.to_csv(out / "summary.csv")
    summary.to_csv(out / "diagnostics_all_parameters.csv")

    metrics.update(
        run=dataclasses.asdict(run), seed=run.seed, sampling=cfg, quick=quick,
        hours_sampling=(t_sampled - t_start) / 3600, hours_total=(time.time() - t_start) / 3600,
        versions=dict(python=sys.version.split()[0], pymc=pm.__version__, pytensor=pytensor.__version__,
                      arviz=az.__version__, numpy=np.__version__, platform=platform.platform(),
                      cxx=pytensor.config.cxx),
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    (out / "done.json").write_text(json.dumps(dict(finished=time.strftime("%Y-%m-%d %H:%M:%S"))), encoding="utf-8")
    c = metrics["convergence"]
    log(f"done: R-hat max {c['rhat_max']:.3f} ({c['rhat_core']:.3f} excluding K-prior planets), "
        f"ESS min {c['ess_bulk_min']:.0f} ({c['ess_core']:.0f}), divergences {c['divergences']}, "
        f"ln L-hat {metrics['ll_hat']:.2f}, BIC {metrics['bic']:.2f}")


if __name__ == "__main__":
    main(sys.argv[1], quick="--quick" in sys.argv[2:])

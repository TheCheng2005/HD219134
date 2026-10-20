"""Post-sampling diagnostics and model-comparison statistics.

Definitions (see README, "What the numbers mean"):
  ln L      log-likelihood of the data (model.datalogp: no priors, no Jacobians; it also contains the
            e < 1 potential, which is below 1e-100 for any e under 0.3)
  ln L-hat  best ln L inside the posterior's support: the best posterior draw, or the
            posterior mode reached from the best draw of each chain, whichever is higher.
            A likelihood-only optimum is NOT used: the light curve does not constrain
            P_rot, so without priors P_rot runs off to ~1e4 d and the GP absorbs planets.
  BIC       k ln n - 2 ln L-hat
  DIC       D_bar + p_D, with D = -2 ln L
"""
import arviz as az
import numpy as np
import pytensor
import scipy.optimize as so
from celerite2.backprop import LinAlgError

KEY_PARAMS = ["P_rot", "period", "K", "m_pl", "ecc", "C_activity", "log_sigma_rv_gp", "rv_jitters",
              "m_star", "r_star"]


def make_point_builder(model):
    """Map natural-scale values {rv name: value} onto the model's value variables.

    Bounded parameters are sampled on a transformed scale. In PyMC 5 the transform is
    model.rvs_to_transforms[rv]; rv.transform is always None (the paper-era bug).
    """
    base_point = model.initial_point()
    mapping = []
    for rv in model.free_RVs:
        transform = model.rvs_to_transforms.get(rv)
        forward = None
        if transform is not None:
            v = rv.type()
            forward = pytensor.function([v], transform.forward(v, *rv.owner.inputs), on_unused_input="ignore")
        mapping.append((rv.name, model.rvs_to_values[rv].name, forward))

    def to_point(natural):
        point = base_point.copy()
        for rv_name, value_name, forward in mapping:
            val = natural[rv_name]
            point[value_name] = val if forward is None else forward(val)
        return point

    return to_point


def posterior_draws(model, trace):
    """Natural-scale draws of every free parameter, sample axis last (chain-major)."""
    post = trace.posterior.stack(sample=("chain", "draw"))
    return {rv.name: post[rv.name].values for rv in model.free_RVs}, post.sizes["sample"]


def _draw(draws, i):
    return {k: d[..., i] for k, d in draws.items()}


def mapping_error(model, trace, to_point, draws, n_check=20):
    """Rebuilding draws must reproduce the log-probability the sampler stored."""
    lp = trace.sample_stats["lp"].stack(sample=("chain", "draw")).values
    f = model.compile_logp()
    idx = np.linspace(0, len(lp) - 1, n_check).astype(int)
    return float(max(abs(f(to_point(_draw(draws, i))) - lp[i]) for i in idx))


def posterior_mode(model, start, scale, maxiter=5000):
    """L-BFGS on the log-posterior from `start`, with parameters rescaled by `scale`."""
    vars_ = model.value_vars
    logp = model.logp()
    f = model.compile_fn([logp, *pytensor.grad(logp, vars_, disconnected_inputs="ignore")],
                         inputs=vars_, on_unused_input="ignore")
    shapes = [np.shape(start[v.name]) for v in vars_]
    sizes = [int(np.prod(s)) for s in shapes]

    def unflatten(x):
        out, i = {}, 0
        for v, s, n in zip(vars_, shapes, sizes):
            out[v.name] = x[i:i + n].reshape(s)
            i += n
        return out

    x0 = np.concatenate([np.ravel(start[v.name]) for v in vars_])

    best = {"lp": -np.inf, "z": np.zeros_like(x0)}  # scipy may return an invalid last trial point

    def neg(z):
        try:
            lp, *g = f(unflatten(x0 + scale * z))
        except LinAlgError:  # GP covariance not factorizable: reject the step
            return 1e30, np.zeros_like(z)
        if lp > best["lp"]:
            best.update(lp=lp, z=z.copy())
        return -lp, -np.concatenate([np.ravel(gi) for gi in g]) * scale

    res = so.minimize(neg, np.zeros_like(x0), jac=True, method="L-BFGS-B",
                      options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-8, "maxcor": 50})
    return unflatten(x0 + scale * best["z"]), str(res.message)


def describe(model, point):
    """Key natural-scale parameters and the LC/RV split of ln L at a value-variable point."""
    names = [n for n in KEY_PARAMS if n in model.named_vars]
    outs = model.replace_rvs_by_values([model[n] for n in names])
    outs += [model.logp(vars=[model["obs_lc"]], sum=True), model.logp(vars=[model["obs_rv"]], sum=True)]
    vals = model.compile_fn(outs, inputs=model.value_vars, on_unused_input="ignore")(point)
    d = {n: np.asarray(v).tolist() for n, v in zip(names, vals)}
    d["ll_lc"], d["ll_rv"] = float(vals[-2]), float(vals[-1])
    return d


def convergence(trace, model, planets, circular=()):
    """R-hat/ESS over all parameters, and over all except the period, phase and K of the
    circular (disputed or injected) planets: when such a planet's K is near 0 those are
    unconstrained and their R-hat says nothing about the rest of the fit."""
    names = [rv.name for rv in model.free_RVs] + ["K", "period"]
    s = az.summary(trace, var_names=names, kind="diagnostics")
    drop = {f"{v}[{planets.index(p)}]" for p in circular for v in ("t0", "log_period", "period", "K", "K_pl")}
    core = s[[i not in drop for i in s.index]]
    ss = trace.sample_stats
    post = trace.posterior
    per_chain = {
        "P_rot": post["P_rot"].median("draw").values.tolist(),
        "period": {p: post["period"][..., j].median("draw").values.tolist() for j, p in enumerate(planets)},
        "K": {p: post["K"][..., j].median("draw").values.tolist() for j, p in enumerate(planets)},
    }
    return dict(
        rhat_max=float(s["r_hat"].max()), ess_bulk_min=float(s["ess_bulk"].min()),
        rhat_core=float(core["r_hat"].max()), ess_core=float(core["ess_bulk"].min()),
        ess_tail_min=float(s["ess_tail"].min()),
        worst=s.sort_values("r_hat", ascending=False).head(6)[["r_hat", "ess_bulk"]].round(3).to_dict("index"),
        divergences=int(ss["diverging"].sum()),
        frac_max_treedepth=float(ss["reached_max_treedepth"].mean()),
        median_steps=float(np.median(ss["n_steps"])),
        per_chain_medians=per_chain,
    ), s


def planet_table(trace, planets):
    post = trace.posterior
    out = {}
    for j, p in enumerate(planets):
        out[p] = {}
        for var in ("period", "K", "m_pl", "ecc"):
            v = post[var][..., j].values.ravel()
            out[p][var] = [float(np.percentile(v, q)) for q in (16, 50, 84)]
        out[p]["K_2.5pct"] = float(np.percentile(post["K"][..., j].values, 2.5))
    return out


def analyze(model, trace, n_obs, planets, circular=(), progress=print):
    to_point = make_point_builder(model)
    draws, n_total = posterior_draws(model, trace)
    n_chain, n_draw = trace.posterior.sizes["chain"], trace.posterior.sizes["draw"]

    err = mapping_error(model, trace, to_point, draws)
    progress(f"mapping check: max |logp(rebuilt) - lp(sampler)| = {err:.2e}")
    if err > 1e-6:
        raise RuntimeError(f"posterior draws do not reproduce the sampler's log-probability (err {err})")

    loglike = model.compile_fn(model.datalogp, inputs=model.value_vars, on_unused_input="ignore")
    ll = np.array([loglike(to_point(_draw(draws, i))) for i in range(n_total)])
    if not np.all(np.isfinite(ll)):
        raise RuntimeError("non-finite log-likelihood at a posterior draw")
    ll_chain = ll.reshape(n_chain, n_draw)  # stacked sample index is chain-major
    progress(f"ln L over {n_total} draws: max {ll.max():.2f}, median {np.median(ll):.2f}")

    flat = lambda p: np.concatenate([np.ravel(p[v.name]) for v in model.value_vars])
    scale = np.std([flat(to_point(_draw(draws, i))) for i in range(0, n_total, max(1, n_total // 500))], axis=0)
    scale[scale == 0] = 1.0

    chains, best = [], None
    for c in range(n_chain):
        i_best = c * n_draw + int(np.argmax(ll_chain[c]))
        mode, msg = posterior_mode(model, to_point(_draw(draws, i_best)), scale)
        ll_mode = float(loglike(mode))
        chains.append(dict(median_ll=float(np.median(ll_chain[c])), max_ll=float(ll_chain[c].max()),
                           mode_ll=ll_mode, mode_msg=msg))
        progress(f"chain {c}: median {chains[-1]['median_ll']:.2f}, best draw {chains[-1]['max_ll']:.2f}, "
                 f"ln L at posterior mode {ll_mode:.2f} ({msg})")
        for cand_ll, cand in ((ll_chain[c].max(), to_point(_draw(draws, i_best))), (ll_mode, mode)):
            if best is None or cand_ll > best[0]:
                best = (float(cand_ll), cand)

    ll_hat, best_point = best
    k = int(sum(np.size(v) for v in model.initial_point().values()))
    d_bar = float(np.mean(-2 * ll))
    d_hat = float(-2 * loglike(to_point({kk: np.median(d, axis=-1) for kk, d in draws.items()})))
    conv, summary = convergence(trace, model, planets, circular)

    return dict(
        k=k, n_obs=int(n_obs), mapping_error=err,
        ll_hat=ll_hat, ll_max_draw=float(ll.max()), ll_median=float(np.median(ll)),
        bic=float(k * np.log(n_obs) - 2 * ll_hat), aic=float(2 * k - 2 * ll_hat),
        dic=float(2 * d_bar - d_hat), p_dic=float(d_bar - d_hat),
        chains=chains, best_fit=describe(model, best_point),
        convergence=conv, planets=planet_table(trace, planets),
    ), summary

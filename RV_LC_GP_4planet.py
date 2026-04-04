import exoplanet as xo
import numpy as np
import arviz as az
from astropy.stats import sigma_clip
import os
import pytensor
import pytensor.tensor as pt
import astropy.units as u
import pymc as pm
import pymc_ext as pmx
import pickle
import matplotlib.pyplot as plt
from celerite2.pymc import GaussianProcess, terms

# --- CONFIGURATION ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# =============================================================================
# 1. DATA LOADING (MULTI-INSTRUMENT)
# =============================================================================
def load_data():
    x_ref = 2450000.

    # --- 1. LIGHT CURVE ---
    print("Loading TESS data...")
    file_paths = [
        'n20191008.TESS.TESS.TIC283722336.S17.0120.SPOC.dat',
        'n20200416.TESS.TESS.TIC283722336.S24.0120.SPOC.dat',
        'n20220930.TESS.TESS.TIC283722336.S57.0120.SPOC.dat',
        'n20240325.TESS.TESS.TIC283722336.S77.0120.SPOC.dat',
        'n20241001.TESS.TESS.TIC283722336.S84.0120.SPOC.dat'
    ]

    all_time, all_flux, all_flux_error = [], [], []
    for file_path in file_paths:
        if not os.path.exists(file_path): continue
        data = np.loadtxt(file_path, comments='#')

        time_raw, flux_raw, err_raw = data[:, 0], data[:, 1], data[:, 2]
        total_raw = len(flux_raw)

        # 1. NaN Removal
        m = np.isfinite(flux_raw) & np.isfinite(time_raw)
        time, flux, flux_error = time_raw[m], flux_raw[m], err_raw[m]
        nans_removed = total_raw - len(flux)

        # 2. Sigma Clipping & High Flux Masks
        mask_clip = sigma_clip(flux, sigma=4, maxiters=3).mask
        mask_high = flux > 1.002
        clean_mask = (~mask_clip) & (~mask_high)

        sigma_removed = np.sum(mask_clip)
        high_removed = np.sum(mask_high)
        total_clean = np.sum(clean_mask)

        print(f"  File: {file_path}")
        print(f"    Raw points:           {total_raw}")
        print(f"    NaNs removed:         {nans_removed}")
        print(f"    4-sigma removed:      {sigma_removed}")
        print(f"    >1.002 flux removed:  {high_removed}")
        print(f"    Final clean points:   {total_clean}")

        all_time.append(time[clean_mask])
        all_flux.append(flux[clean_mask])
        all_flux_error.append(flux_error[clean_mask])
    # Concatenate the lists into flat arrays and apply scaling
    x_full = np.concatenate(all_time) - x_ref
    y_full = (np.concatenate(all_flux) - 1) * 1e3
    yerr_full = np.concatenate(all_flux_error) * 1e3

    # Add the 1e-6 jitter
    x, y, yerr = x_full, y_full, yerr_full + 1e-6

    # Sort chronologically
    idx = np.argsort(x)
    x, y, yerr = x[idx], y[idx], yerr[idx]

    # --- 2. RV DATA & SHK ---
    print("Loading RV & SHK data...")
    rv_x_list, rv_y_list, rv_yerr_list, rv_id_list = [], [], [], []
    shk_list = []
    rv_medians = []

    instruments = [
        ("HARPS",      "HD219134.HARPS.SHK.rv",         0),
        ("HJST",       "HD219134.HJST_Phase3.rv",       1),
        ("HIRES-post", "HD219134.HARADA_HIRES-post.rv", 2),
        ("Levy",       "HD219134.APF.Rosenthal2021.rv", 3),
        ("Hamilton",   "HD219134.Hamilton.Howard2016a1994.rv", 4)
    ]

    inst_names = []

    for name, fname, inst_id in instruments:
        if os.path.exists(fname):
            print(f"  Found {name} ({fname})")
            data = np.loadtxt(fname, comments='#')

            t_inst = data[:, 0] - x_ref
            y_inst = data[:, 1]
            yerr_inst = data[:, 2]
            print(f"The {name} data set has a length of {len(t_inst)}")

            rv_medians.append(np.median(y_inst))

            # --- Strict SHK loading (No Interpolation) ---
            if name == "HARPS" and data.shape[1] >= 4:
                s_val = data[:, 3]
                mask = np.isfinite(t_inst) & np.isfinite(y_inst) & np.isfinite(yerr_inst) & np.isfinite(s_val)
                shk_list.append(s_val[mask])
            else:
                mask = np.isfinite(t_inst) & np.isfinite(y_inst) & np.isfinite(yerr_inst)
                shk_list.append(np.zeros(np.sum(mask)))

            rv_x_list.append(t_inst[mask])
            rv_y_list.append(y_inst[mask])
            rv_yerr_list.append(yerr_inst[mask])
            rv_id_list.append(np.full(np.sum(mask), inst_id, dtype=int))
            inst_names.append(name)

        else:
            print(f"  Warning: {name} file ({fname}) not found.")
            rv_medians.append(0.0)

    if rv_x_list:
        x_rv = np.concatenate(rv_x_list)
        y_rv = np.concatenate(rv_y_list)
        yerr_rv = np.concatenate(rv_yerr_list)
        id_rv = np.concatenate(rv_id_list)
        shk_rv = np.concatenate(shk_list)

        sort_rv = np.argsort(x_rv)
        x_rv = x_rv[sort_rv]
        y_rv = y_rv[sort_rv]
        yerr_rv = yerr_rv[sort_rv]
        id_rv = id_rv[sort_rv]
        shk_rv = shk_rv[sort_rv]
    else:
        raise FileNotFoundError("No RV files found. ERROR")

    # Standardize SHK only for HARPS points; keep zeros as zeros for other instruments
    print("Standardizing SHK for HARPS...")
    shk_rv_std = np.zeros_like(shk_rv)
    harps_mask = (id_rv == 0)

    if np.any(harps_mask):
        shk_harps = shk_rv[harps_mask]
        shk_mean = np.mean(shk_harps)
        shk_std_val = np.std(shk_harps)
        if shk_std_val == 0: shk_std_val = 1.0
        shk_rv_std[harps_mask] = (shk_harps - shk_mean) / shk_std_val

    return x, y, yerr, x_rv, y_rv, yerr_rv, id_rv, inst_names, rv_medians, shk_rv_std

# =============================================================================
# 2. MODEL BUILDING
# =============================================================================
nplanet = 4
ntransit = 2
x_ref = 2450000.

def build_model(x, y, yerr, x_rv, y_rv, yerr_rv, id_rv, inst_names, rv_medians, shk_rv_std):

    n_inst = len(inst_names)
    print(f"Building model for {n_inst} RV instruments.")

    # --- Lit Values ---
    periods_lit = np.array([3.0929385, 6.765047, 46.73, 2180.0])
    log_periods_lit = np.log10(periods_lit)
    log_periods_sigma_lit = np.array([0.0001, 0.0001, 0.05, 0.1])

    t0s_mu = np.array([
        2459263.91706 - x_ref,
        2459435.90931 - x_ref,
        2456744.56 - x_ref,
        2456840.0 - x_ref
    ])
    t0s_sigma = np.array([0.05, 0.05, 0.35, 48.0])

    mass_pl_mu_lit = np.log10([4.3, 4, 14.6, 146.0])
    mass_pl_sigma_lit = np.array([0.2, 0.2, 0.3, 0.4])

    depths_ppt_lit = np.array([0.358, 0.318])
    impact_p_lit = np.array([0.9171, 0.801])

    ecc_mu_lit = np.array([0.0, 0.103, 0.2, 0.2])
    ecc_sigma_lit = np.array([0.1, 0.1, 0.5, 0.5])
    ecc_init = np.maximum(ecc_mu_lit, 0.05)

    texp = 120./86400.

    with pm.Model() as model:

        # --- Stellar ---
        mean_flux = pm.Normal("mean_flux", mu=0.0, sigma=5)
        log_jitter_lc = pm.Normal("log_jitter_lc", mu=np.log(np.nanstd(y)), sigma=2.0)

        # Limb Darkening using Normal Priors on q1, q2
        q1 = pm.Normal("q1", mu=0.4267, sigma=0.0924)
        q2 = pm.Normal("q2", mu=0.3705, sigma=0.0300)

        # Transform Kipping (2013) q1, q2 parameters back to u1, u2
        u1 = pm.Deterministic("u1", 2 * pt.sqrt(q1) * q2)
        u2 = pm.Deterministic("u2", pt.sqrt(q1) * (1 - 2 * q2))
        u_star = pt.stack([u1, u2])

        star = xo.LimbDarkLightCurve(u_star)

        # Ashley Interferometry Data
        m_star = pm.Normal("m_star", mu=0.763, sigma=0.2)
        r_star = pm.Normal("r_star", mu=0.792, sigma=0.008)
        # Li Yaguang, et.al (2025)
        #m_star = pm.Normal("m_star", mu=0.763, sigma=0.0212)
        #r_star = pm.Normal("r_star", mu=0.748, sigma=0.0073)

        # --- Planets ---
        t0 = pm.Normal("t0", mu=t0s_mu, sigma=t0s_sigma, shape=nplanet, initval=t0s_mu)
        log_period = pm.Normal("log_period", mu=log_periods_lit, sigma=log_periods_sigma_lit, shape=nplanet, initval=log_periods_lit)
        period = pm.Deterministic("period", 10**log_period)

        log_m_pl = pm.Normal("log_m_pl", mu=mass_pl_mu_lit, sigma=mass_pl_sigma_lit, shape=nplanet)
        m_pl = pm.Deterministic("m_pl", 10**log_m_pl)

        # Eccentricity (Updated Upper Bound to 0.99)
        ecc = pm.TruncatedNormal("ecc", mu=ecc_mu_lit, sigma=ecc_sigma_lit, lower=0.0, upper=0.99, shape=nplanet, initval=ecc_init)
        omega = pm.Normal("omega", mu=0.0, sigma=3.0, shape=nplanet)

        # --- Geometry---
        b_transiting = pm.Uniform("b_transiting", lower=0.0, upper=1.2, shape=ntransit, initval=impact_p_lit)
        b_nontransiting = pt.zeros(nplanet - ntransit)
        b_all = pm.Deterministic("b_all", pt.concatenate([b_transiting, b_nontransiting]))

        log_depth = pm.Normal("log_depth", mu=np.log10(depths_ppt_lit), sigma=0.5, shape=ntransit)
        ror = pm.Deterministic("ror", star.get_ror_from_approx_transit_depth(1e-3 * 10**log_depth, b_transiting))
        r_pl = pm.Deterministic("r_pl", ror * r_star)

        # --- Likelihoods ---
        # 1. Light Curve
        orbit_transit = xo.orbits.KeplerianOrbit(
            r_star=r_star, m_star=m_star,
            period=period[:ntransit], t0=t0[:ntransit], b=b_transiting,
            m_planet=xo.units.with_unit(m_pl[:ntransit], u.Mearth),
            ecc=ecc[:ntransit], omega=omega[:ntransit]
        )
        light_curves = star.get_light_curve(orbit=orbit_transit, r=r_pl, t=x, texp=texp) * 1e3
        lc_model = pt.sum(light_curves, axis=-1) + mean_flux

        # --- Shared Stellar Rotation Period ---
        log_P_rot = pm.Normal("log_P_rot", mu=np.log(22.6), sigma=0.1)
        P_rot = pm.Deterministic("P_rot", pt.exp(log_P_rot))

        # --- LC GP (RotationTerm) ---
        log_sigma_lc = pm.Normal("log_sigma_lc", mu=np.log(np.nanstd(y)), sigma=2.0)
        log_Q0_lc = pm.Normal("log_Q0_lc", mu=1.0, sigma=1.0)
        log_dQ_lc = pm.Normal("log_dQ_lc", mu=0.0, sigma=1.0)
        f_rot_lc = pm.Uniform("f_rot_lc", lower=0.05, upper=1.0)

        kernel_lc = terms.RotationTerm(
            sigma=pt.exp(log_sigma_lc),
            period=P_rot,
            Q0=pt.exp(log_Q0_lc),
            dQ=pt.exp(log_dQ_lc),
            f=f_rot_lc
        )

        gp_lc = GaussianProcess(
            kernel_lc,
            t=x,
            yerr=pt.sqrt(yerr**2 + pt.exp(2*log_jitter_lc)),
            mean=lc_model
        )
        gp_lc.marginal("obs_lc", observed=y)

        # --- RV GP (RotationTerm) ---
        # Give RV its own amplitudes and quality parameters, but share P_rot
        log_sigma_rv_gp = pm.Normal("log_sigma_rv_gp", mu=np.log(2.0), sigma=1.0)
        log_Q0_rv = pm.Normal("log_Q0_rv", mu=1.0, sigma=1.0)
        log_dQ_rv = pm.Normal("log_dQ_rv", mu=0.0, sigma=1.0)
        f_rot_rv = pm.Uniform("f_rot_rv", lower=0.05, upper=1.0)

        kernel_rv = terms.RotationTerm(
            sigma=pt.exp(log_sigma_rv_gp),
            period=P_rot,  # Shared Period
            Q0=pt.exp(log_Q0_rv),
            dQ=pt.exp(log_dQ_rv),
            f=f_rot_rv
        )

        # --- RV Orbits & Activity ---
        orbit_rv = xo.orbits.KeplerianOrbit(
            r_star=r_star, m_star=m_star,
            period=period, t0=t0, b=b_all,
            m_planet=xo.units.with_unit(m_pl, u.Mearth),
            ecc=ecc, omega=omega
        )
        rv_model_sum = pt.sum(orbit_rv.get_radial_velocity(x_rv), axis=-1)
        rv_offsets = pm.Normal("rv_offsets", mu=0.0, sigma=25000.0, shape=n_inst, initval=rv_medians)
        rv_jitters = pm.HalfNormal("rv_jitters", sigma=10.0, shape=n_inst)

        # Activity Model (Strict linear detrending matching exact RV timestamps)
        C_activity = pm.Normal("C_activity", mu=0.0, sigma=5.0, initval=0.0)
        activity_model = C_activity * shk_rv_std
        total_rv_mean = rv_model_sum + rv_offsets[id_rv] + activity_model

        gp_rv = GaussianProcess(
            kernel_rv,
            t=x_rv,
            yerr=pt.sqrt(yerr_rv**2 + rv_jitters[id_rv]**2 + 1e-4),
            mean=total_rv_mean)
        gp_rv.marginal("obs_rv", observed=y_rv)

        return model
# =============================================================================
# 3. ANALYSIS & SAVING
# =============================================================================
def print_map_values(map_sol, inst_names):
    print("\n" + "="*40)
    print("      MAP SOLUTION")
    print("="*40)
    def get_val(key):
        val = map_sol[key]
        return val if np.shape(val) else np.array([val])

    per = get_val('period')
    t0 = get_val('t0')
    ecc = get_val('ecc')
    mass = get_val('m_pl')
    ror = get_val('ror')
    P_rot = get_val('P_rot')[0]

    print(f"{'Pl':<4} {'Period':<10} {'Ecc':<8} {'Mass':<8}")
    for i in range(len(per)):
        print(f"{i:<4} {per[i]:<10.4f} {ecc[i]:<8.3f} {mass[i]:<8.3f}")

    print("-" * 30)
    offsets = get_val('rv_offsets')
    jitters = get_val('rv_jitters')
    for i, name in enumerate(inst_names):
        print(f"{name:<10} Offset: {offsets[i]:<10.1f} Jitter: {jitters[i]:<5.2f} P_rot: {P_rot:.4f}")
    print("="*40 + "\n")

def check_gradient(model, point):
    print("\n=== DEBUG: Gradient Check ===")
    with model:
        try:
            vars_list = model.value_vars
            names = [v.name for v in vars_list]
            grads_sym = pm.gradient(model.logp(), vars_list)

            print("Compiling raw gradient function...")
            fn = pytensor.function(
                inputs=vars_list,
                outputs=grads_sym,
                on_unused_input='ignore'
            )

            input_values = []
            for v in vars_list:
                name = v.name
                if name in point:
                    input_values.append(point[name])
                else:
                    input_values.append(np.zeros(v.eval().shape))

            print("Calculating gradients...")
            grad_values = fn(*input_values)

            print(f"\n{'Variable Name':<35} | {'Gradient Norm':<15}")
            print("-" * 55)

            max_grad = 0
            max_name = ""

            for name, val in zip(names, grad_values):
                gnorm = np.linalg.norm(val)
                print(f"{name:<35} | {gnorm:<15.2e}")
                if gnorm > max_grad:
                    max_grad = gnorm
                    max_name = name

            print("-" * 55)
            print(f"Max Gradient: {max_grad:.2e} on '{max_name}'")
            if max_grad > 1e4:
                print(f"\n>>> CRITICAL WARNING: High Gradients Detected with '{max_name}'.")

        except Exception as e:
            print(f"DEBUGGER FAILED: {e}")

def get_aic_bic_metric(model, trace, n_obs, n_samples=2000):
    import pytensor.tensor as pt
    from tqdm import tqdm
    import numpy as np
    import matplotlib.pyplot as plt

    print("\n" + "="*40)
    print("   MEDIAN AIC/BIC & LOG-LIKELIHOOD")
    print("="*40)

    with model:
        try:
            loglike_func = model.compile_fn(model.datalogp)
        except Exception as e:
            print(f"Error compiling datalogp: {e}")
            return None

    posterior = trace.posterior.stack(sample=("chain", "draw"))
    n_total = posterior.sizes["sample"]
    indices = np.random.choice(n_total, size=n_samples, replace=False) if n_total > n_samples else np.arange(n_total)

    log_likelihoods = []
    base_point = model.initial_point()

    print(f"Evaluating log-likelihood for {len(indices)} posterior samples...")
    for idx in tqdm(indices, desc="Calculating Log-Likelihoods"):
        point = base_point.copy()

        for rv in model.free_RVs:
            rv_name = rv.name
            if rv_name not in posterior.data_vars:
                continue

            val_natural = posterior[rv_name].isel(sample=idx).values
            value_var = model.rvs_to_values[rv]
            value_name = value_var.name
            transform = getattr(rv, 'transform', None)

            if transform is not None:
                try:
                    val_transformed = transform.forward(pt.as_tensor(val_natural)).eval()
                    point[value_name] = val_transformed
                except Exception:
                    pass
            else:
                point[value_name] = val_natural

        try:
            ll = loglike_func(point)
            if np.isfinite(ll):
                log_likelihoods.append(ll)
        except Exception:
            pass

    ll_array = np.array(log_likelihoods)
    if len(ll_array) == 0:
        return None

    median_log_likelihood = np.median(ll_array)

    k = sum(model.initial_point()[var.name].size for var in model.value_vars)
    aic = 2 * k - 2 * median_log_likelihood
    bic = k * np.log(n_obs) - 2 * median_log_likelihood

    print(f"\n" + "-"*40)
    print(f"Parameters (k):          {k}")
    print(f"Observations (n):        {n_obs}")
    print(f"Median Log-Likelihood:   {median_log_likelihood:.4f}")
    print(f"\nAIC (2k - 2*logL):       {aic:.4f}")
    print(f"BIC (k*log(n) - 2*logL): {bic:.4f}")
    print("="*40 + "\n")

    return {
        'aic': aic,
        'bic': bic,
        'k': k,
        'n_obs': n_obs,
        'median_log_likelihood': median_log_likelihood
    }

def get_dic_score(model, trace, samples=2000):
    import numpy as np
    from tqdm import tqdm

    print(f"--- Computing DIC (Samples: {samples}) ---")
    with model:
        logp_func = model.compile_logp()

    posterior = trace.posterior.stack(sample=("chain", "draw"))
    n_total = posterior.sizes["sample"]
    indices = np.random.choice(n_total, size=samples, replace=False) if n_total > samples else np.arange(n_total)

    base_point = model.initial_point()
    trace_vars = list(trace.posterior.data_vars)
    relevant_vars = [v for v in trace_vars if v in base_point]
    data_cache = {v: posterior[v].values for v in relevant_vars}

    deviances = []
    for i in tqdm(indices, desc="Calculating Deviance"):
        point = base_point.copy()
        for var in relevant_vars:
            point[var] = data_cache[var][..., i]
        try:
            deviances.append(-2 * logp_func(point))
        except Exception:
            pass

    d_bar = np.mean(deviances)
    median_point = base_point.copy()
    for var in relevant_vars:
        median_point[var] = np.median(data_cache[var], axis=-1)

    d_hat = -2 * logp_func(median_point)
    p_d = d_bar - d_hat
    dic = d_bar + p_d

    print(f"Mean Deviance (D_bar):     {d_bar:.4f}")
    print(f"Effective Params (p_D):    {p_d:.4f}")
    print(f"DIC:                       {dic:.4f}")
    print("="*40 + "\n")
    return dic, p_d

def run_analysis(model, trace, map_sol, x, y, x_rv, y_rv, yerr_rv, id_rv, inst_names):
    print("\n--- Running Analysis ---")
    trace.to_netcdf("rv_lc_gp_4planet_trace.nc")
    with open("rv_lc_gp_4planet_map.pkl", "wb") as f:
        pickle.dump(map_sol, f)

    summ = az.summary(trace, var_names=["period", "m_pl", "ecc", "rv_offsets", "log_P_rot"])
    print(summ)

# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    x, y, yerr, x_rv, y_rv, yerr_rv, id_rv, inst_names, rv_medians, shk_rv_std = load_data()
    print(f"Loaded {len(x)} LC points, {len(x_rv)} RV points from {inst_names}")

    model = build_model(x, y, yerr, x_rv, y_rv, yerr_rv, id_rv, inst_names, rv_medians, shk_rv_std)

    with model:
        print("\n=== MAP Optimization (Iterative) ===")
        start_pt = model.initial_point()

        print("1. Optimizing Phase (Period, T0)...")
        map_sol = pmx.optimize(start=start_pt, vars=[model['t0'], model['log_period']])

        print("2. Optimizing Baseline (Offsets, Jitters, Mean Flux, Activity, GP)...")
        vars_step2 = [
            model['rv_offsets'], model['mean_flux'], model['log_jitter_lc'], model['rv_jitters'],
            model['C_activity']
        ]
        map_sol = pmx.optimize(start=map_sol, vars=vars_step2)

        print("3. Optimizing GPs (LC and RV jointly sharing P_rot)...")
        vars_gp = [
            model["log_P_rot"],
            model["log_Q0_lc"], model["log_dQ_lc"], model["f_rot_lc"], model["log_sigma_lc"],
            model["log_Q0_rv"], model["log_dQ_rv"], model["f_rot_rv"], model["log_sigma_rv_gp"]
        ]
        map_sol = pmx.optimize(start=map_sol, vars=vars_gp)

        print("4. Optimizing Geometry (b, depth, r_star)...")
        vars_step3 = [model['b_transiting'], model['log_depth'], model['r_star']]
        map_sol = pmx.optimize(start=map_sol, vars=vars_step3)

        print("5. Optimizing Orbits (Ecc, Omega, Mass)...")
        vars_step4 = [model['ecc'], model['omega'], model['log_m_pl']]
        map_sol = pmx.optimize(start=map_sol, vars=vars_step4)

        print("6. Optimizing Full Model...")
        map_sol = pmx.optimize(start=map_sol)

        print("\n=== Checking Initial Gradients ===")
        check_gradient(model, map_sol)
        print_map_values(map_sol, inst_names)

        print("\n=== Sampling ===")
        trace = pm.sample(
            tune=1600,
            draws=1600,
            initvals=map_sol,
            init="adapt_full",
            chains=4,
            cores=4,
            target_accept=0.98,
            return_inferencedata=True,
        )

    run_analysis(model, trace, map_sol, x, y, x_rv, y_rv, yerr_rv, id_rv, inst_names)

    print("\n" + "="*60)
    print("COMPUTING MODEL COMPARISON METRICS")
    print("="*60)

    n_obs = len(x) + len(x_rv)
    print(f"Total Observations: {len(x)} (LC) + {len(x_rv)} (RV) = {n_obs}")

    get_dic_score(model, trace)
    results = get_aic_bic_metric(model, trace, n_obs)

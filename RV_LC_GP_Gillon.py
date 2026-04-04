#Gillon models b, c, f, d
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

    # --- 1. LIGHT CURVE (TESS) ---
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
        time, flux, flux_error = data[:, 0], data[:, 1], data[:, 2]
        m = np.isfinite(flux) & np.isfinite(time)
        time, flux, flux_error = time[m], flux[m], flux_error[m]

        mask_clip = sigma_clip(flux, sigma=4, maxiters=3).mask
        mask_high = flux > 1.002
        clean_mask = (~mask_clip) & (~mask_high)

        all_time.append(time[clean_mask])
        all_flux.append(flux[clean_mask])
        all_flux_error.append(flux_error[clean_mask])

    if not all_time:
        print("Warning: Data files not found. Using Mock Data.")
        t_mock = np.linspace(2459263 - x_ref - 5, 2459435 - x_ref + 5, 5000)
        x_full = t_mock
        y_full = np.random.normal(0, 0.1, len(t_mock))
        yerr_full = np.ones_like(t_mock) * 0.1
    else:
        x_full = np.concatenate(all_time) - x_ref
        y_full = (np.concatenate(all_flux) - 1) * 1e3
        yerr_full = np.concatenate(all_flux_error) * 1e3

    # Masking (Transits + Baseline)
    transit_t0s = np.array([2459263.91706 - x_ref, 2459435.90931 - x_ref])
    transit_periods = np.array([3.0929385, 6.7650482])
    transit_durations = np.array([0.2, 0.25])

    near_transit_mask = np.zeros_like(x_full, dtype=bool)
    for t0, per, dur in zip(transit_t0s, transit_periods, transit_durations):
        phase = (x_full - t0 + 0.5 * per) % per - 0.5 * per
        near_transit_mask |= (np.abs(phase) < (0.6 * dur))

    baseline_mask = np.zeros_like(x_full, dtype=bool)
    baseline_mask[::30] = True
    final_mask = near_transit_mask | baseline_mask

    #x, y, yerr = x_full[final_mask], y_full[final_mask], yerr_full[final_mask] + 1e-6
    x, y, yerr = x_full, y_full, yerr_full + 1e-6
    idx = np.argsort(x)
    x, y, yerr = x[idx], y[idx], yerr[idx]

    # --- 2. RV DATA (MULTI-INSTRUMENT) & SHK ---
    print("Loading RV & SHK data...")
    rv_x_list, rv_y_list, rv_yerr_list, rv_id_list = [], [], [], []
    rv_medians = [] # Store medians for smart initialization
    shk_x, shk_y = [], []

    # === USER: UPDATE FILENAMES HERE ===
    # Format: (Name, Filename, ID_Number)
    instruments = [
        ("HARPS",      "HD219134.HARPS.SHK.rv",         0),  # Absolute RVs (~ -18400)
        ("HJST",       "HD219134.HJST_Phase3.rv",       1),  # Relative RVs (~ 0)
        ("HIRES-post", "HD219134.HARADA_HIRES-post.rv", 2),  # Harada extension
        ("Levy",       "HD219134.HARADA_Levy.rv",       3),  # Harada extension
        ("Hamilton",   "HD219134.HARADA_Hamilton.rv",   4)   # Harada extension
    ]

    inst_names = []

    for name, fname, inst_id in instruments:
        if os.path.exists(fname):
            print(f"  Found {name} ({fname})")
            data = np.loadtxt(fname, comments='#')

            t_inst = data[:, 0] - x_ref
            y_inst = data[:, 1]
            yerr_inst = data[:, 2]

            rv_medians.append(np.median(y_inst))

            mask = np.isfinite(t_inst) & np.isfinite(y_inst) & np.isfinite(yerr_inst)
            rv_x_list.append(t_inst[mask])
            rv_y_list.append(y_inst[mask])
            rv_yerr_list.append(yerr_inst[mask])
            rv_id_list.append(np.full(np.sum(mask), inst_id, dtype=int))
            inst_names.append(name)

            # --- SHK LOADING (from test.py) ---
            if name == "HARPS" and data.shape[1] >= 4:
                s_val = data[:, 3]
                mask_s = np.isfinite(s_val) & np.isfinite(t_inst)
                shk_x = t_inst[mask_s]
                shk_y = s_val[mask_s]
        else:
            print(f"  Warning: {name} file ({fname}) not found.")
            rv_medians.append(0.0)

    if rv_x_list:
        x_rv = np.concatenate(rv_x_list)
        y_rv = np.concatenate(rv_y_list)
        yerr_rv = np.concatenate(rv_yerr_list)
        id_rv = np.concatenate(rv_id_list)

        sort_rv = np.argsort(x_rv)
        x_rv = x_rv[sort_rv]
        y_rv = y_rv[sort_rv]
        yerr_rv = yerr_rv[sort_rv]
        id_rv = id_rv[sort_rv]
    else:
        print("  No RV files found. ERROR")

    # --- Process SHK ---
    if len(shk_x) == 0:
        shk_x = x_rv
        shk_y = np.zeros_like(x_rv)

    idx_s = np.argsort(shk_x)
    shk_x, shk_y = shk_x[idx_s], shk_y[idx_s]

    print("Standardizing SHK (Zero mean, Unit variance)...")
    shk_mean = np.mean(shk_y)
    shk_std_val = np.std(shk_y)
    if shk_std_val == 0: shk_std_val = 1.0
    shk_y_std = (shk_y - shk_mean) / shk_std_val

    return x, y, yerr, x_rv, y_rv, yerr_rv, id_rv, inst_names, rv_medians, shk_x, shk_y_std

# =============================================================================
# 2. MODEL BUILDING
# =============================================================================
nplanet = 4
ntransit = 2
x_ref = 2450000.

def build_model(x, y, yerr, x_rv, y_rv, yerr_rv, id_rv, inst_names, rv_medians, shk_x, shk_y):

    n_inst = len(inst_names)
    print(f"Building model for {n_inst} RV instruments.")

    # Interpolate SHK to RV timestamps
    shk_at_rv_val = np.interp(x_rv, shk_x, shk_y)

    # --- Lit Values ---
    periods_lit = np.array([3.0929385, 6.765047, 22.72, 46.73])
    log_periods_lit = np.log10(periods_lit)
    log_periods_sigma_lit = np.array([0.0001, 0.0001, 0.05, 0.05])

    t0s_mu = np.array([
        2459263.91706 - x_ref,
        2459435.90931 - x_ref,
        2455233.17 - x_ref,
        2456744.56 - x_ref
    ])
    t0s_sigma = np.array([0.05, 0.05, 0.7, 0.35])

    mass_pl_mu_lit = np.log10([4.3, 4, 7.3, 14.6])
    mass_pl_sigma_lit = np.array([0.2, 0.2, 0.3, 0.3])

    depths_ppt_lit = np.array([0.358, 0.318])
    impact_p_lit = np.array([0.9171, 0.801])

    ecc_mu_lit = np.array([0.0, 0.103, 0.2, 0.1])
    ecc_sigma_lit = np.array([0.1, 0.1, 0.5, 0.5])
    ecc_init = np.maximum(ecc_mu_lit, 0.05)

    u_star_lit = [0.426, 0.207]
    texp = 120./86400.

    with pm.Model() as model:

        # --- Stellar ---
        mean_flux = pm.Normal("mean_flux", mu=0.0, sigma=5)
        log_jitter_lc = pm.Normal("log_jitter_lc", mu=np.log(np.nanstd(y)), sigma=2.0)

        # We use QuadLimbDark but keep it frozen initially by not including in step 1-3 vars
        u_star = xo.QuadLimbDark("u_star", initval=u_star_lit)
        star = xo.LimbDarkLightCurve(u_star)

        # Li Yaguang, et.al (2025)
        m_star = pm.Normal("m_star", mu=0.763, sigma=0.0212)
        r_star = pm.Normal("r_star", mu=0.748, sigma=0.0073)

        # --- Planets ---
        t0 = pm.Normal("t0", mu=t0s_mu, sigma=t0s_sigma, shape=nplanet, initval=t0s_mu)
        # Fit planet period separately for further planets
        log_period = pm.Normal("log_period", mu=log_periods_lit, sigma=log_periods_sigma_lit, shape=nplanet, initval=log_periods_lit)
        period = pm.Deterministic("period", 10**log_period)

        log_m_pl = pm.Normal("log_m_pl", mu=mass_pl_mu_lit, sigma=mass_pl_sigma_lit, shape=nplanet)
        m_pl = pm.Deterministic("m_pl", 10**log_m_pl)

        # Eccentricity
        ecc = pm.TruncatedNormal("ecc", mu=ecc_mu_lit, sigma=ecc_sigma_lit, lower=0.0, upper=0.6, shape=nplanet, initval=ecc_init)
        omega = pm.Normal("omega", mu=0.0, sigma=3.0, shape=nplanet)

        # --- Geometry (Safe Bounds) ---
        b_transiting = pm.Uniform("b_transiting", lower=0.0, upper=1.2, shape=ntransit, initval=impact_p_lit)
        b_nontransiting = pt.zeros(nplanet - ntransit)
        b_all = pm.Deterministic("b_all", pt.concatenate([b_transiting, b_nontransiting]))

        log_depth = pm.Normal("log_depth", mu=np.log10(depths_ppt_lit), sigma=0.5, shape=ntransit)
        ror = pm.Deterministic("ror", star.get_ror_from_approx_transit_depth(1e-3 * 10**log_depth, b_transiting))
        r_pl = pm.Deterministic("r_pl", ror * r_star)

        # --- Likelihoods ---
        # 1. Light Curve (UPDATED: Rotation GP ONLY)
        orbit_transit = xo.orbits.KeplerianOrbit(
            r_star=r_star, m_star=m_star,
            period=period[:ntransit], t0=t0[:ntransit], b=b_transiting,
            m_planet=xo.units.with_unit(m_pl[:ntransit], u.Mearth),
            ecc=ecc[:ntransit], omega=omega[:ntransit]
        )
        light_curves = star.get_light_curve(orbit=orbit_transit, r=r_pl, t=x, texp=texp) * 1e3
        lc_model = pt.sum(light_curves, axis=-1) + mean_flux

        # --- NEW: LC GP (Rotation Only) ---

        # Period of rotation (P_rot)
        log_P_rot = pm.Normal("log_P_rot", mu=np.log(23.0), sigma=1)
        P_rot = pm.Deterministic("P_rot", pt.exp(log_P_rot))

        # Amplitude (sigma)
        log_sigma_lc = pm.Normal("log_sigma_lc", mu=np.log(np.nanstd(y)), sigma=2.0)

        # Quality factor (Q0) and delta Q (dQ)
        log_Q0_lc = pm.Normal("log_Q0_lc", mu=1.0, sigma=1.0)
        log_dQ_lc = pm.Normal("log_dQ_lc", mu=0.0, sigma=1.0)
        f_rot_lc = pm.Uniform("f_rot_lc", lower=0.1, upper=1.0)

        # The Kernel (Pure Rotation)
        kernel_lc = terms.RotationTerm(
            sigma=pt.exp(log_sigma_lc),
            period=P_rot,
            Q0=pt.exp(log_Q0_lc),
            dQ=pt.exp(log_dQ_lc),
            f=f_rot_lc
        )

        # GP Likelihood
        # Note: We still keep the jitter term in the noise model
        gp_lc = GaussianProcess(
            kernel_lc,
            t=x,
            yerr=pt.sqrt(yerr**2 + pt.exp(2*log_jitter_lc)),
            mean=lc_model
        )
        gp_lc.marginal("obs_lc", observed=y)
        # ----------------------------------------------------

        # --- RV (GP + Activity + Orbits) ---
        orbit_rv = xo.orbits.KeplerianOrbit(
            r_star=r_star, m_star=m_star,
            period=period, t0=t0, b=b_all,
            m_planet=xo.units.with_unit(m_pl, u.Mearth),
            ecc=ecc, omega=omega
        )
        rv_model_sum = pt.sum(orbit_rv.get_radial_velocity(x_rv), axis=-1)
        rv_offsets = pm.Normal("rv_offsets", mu=0.0, sigma=25000.0, shape=n_inst, initval=rv_medians)
        rv_jitters = pm.HalfNormal("rv_jitters", sigma=10.0, shape=n_inst)

        # Activity Model (Linear detrending against SHK)
        C_activity = pm.Normal("C_activity", mu=0.0, sigma=5.0, initval=0.0)
        activity_model = C_activity * shk_at_rv_val
        total_rv_mean = rv_model_sum + rv_offsets[id_rv] + activity_model

        # GP for Red Noise in RV
        log_sigma_rv_gp = pm.Normal("log_sigma_rv_gp", mu=np.log(2.0), sigma=1.0)
        log_rho_rv_gp = pm.Normal("log_rho_rv_gp", mu=np.log(10.0), sigma=1.0)

        kernel_rv = terms.SHOTerm(sigma=pt.exp(log_sigma_rv_gp), rho=pt.exp(log_rho_rv_gp), Q=1/3.0)

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

    print(f"{'Pl':<4} {'Period':<10} {'Ecc':<8} {'Mass':<8}")
    for i in range(len(per)):
        print(f"{i:<4} {per[i]:<10.4f} {ecc[i]:<8.3f} {mass[i]:<8.3f}")

    print("-" * 30)
    offsets = get_val('rv_offsets')
    jitters = get_val('rv_jitters')
    for i, name in enumerate(inst_names):
        print(f"{name:<10} Offset: {offsets[i]:<10.1f} Jitter: {jitters[i]:<5.2f}")
    print("="*40 + "\n")

def check_gradient(model, point):
    """
    Robust gradient checker using RAW PyTensor functions.
    Bypasses PyMC Wrapper errors.
    """
    print("\n=== DEBUG: Detailed Gradient Check ===")

    with model:
        try:
            # 1. Get the value variables (parameters the optimizer/sampler sees)
            vars_list = model.value_vars
            names = [v.name for v in vars_list]

            # 2. Calculate Symbolic Gradients of logp
            grads_sym = pm.gradient(model.logp(), vars_list)

            # 3. Compile a RAW PyTensor function
            print("Compiling raw gradient function...")
            fn = pytensor.function(
                inputs=vars_list,
                outputs=grads_sym,
                on_unused_input='ignore'
            )

            # 4. Map the 'point' dictionary to the ordered list of values
            input_values = []
            for v in vars_list:
                name = v.name
                if name in point:
                    input_values.append(point[name])
                else:
                    print(f"WARNING: {name} not found in point dictionary! Using zeros.")
                    input_values.append(np.zeros(v.eval().shape))

            # 5. Call the function
            print("Calculating gradients...")
            grad_values = fn(*input_values)

            # 6. Report
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
            import traceback
            traceback.print_exc()

def get_aic_bic_metric(model, trace, n_obs, n_samples=2000):
    """
    Computes AIC/BIC using the MEDIAN LOG-LIKELIHOOD from posterior samples.
    Bypasses PyMC graph bugs by manually transforming variables for each draw.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pytensor.tensor as pt
    from tqdm import tqdm

    print("\n" + "="*40)
    print("   MEDIAN AIC/BIC & LOG-LIKELIHOOD")
    print("="*40)

    with model:
        # Compile the likelihood function once (no priors)
        try:
            loglike_func = model.compile_fn(model.datalogp)
        except Exception as e:
            print(f"Error compiling datalogp: {e}")
            return None

    # Stack the trace to easily pull random samples
    posterior = trace.posterior.stack(sample=("chain", "draw"))
    n_total = posterior.sizes["sample"]

    if n_total > n_samples:
        indices = np.random.choice(n_total, size=n_samples, replace=False)
    else:
        indices = np.arange(n_total)

    log_likelihoods = []
    base_point = model.initial_point()

    print(f"Evaluating log-likelihood for {len(indices)} posterior samples...")
    for idx in tqdm(indices, desc="Calculating Log-Likelihoods"):
        point = base_point.copy()

        # --- EXACT FALLBACK FROM YOUR ORIGINAL SCRIPT ---
        for rv in model.free_RVs:
            rv_name = rv.name
            if rv_name not in posterior.data_vars:
                continue

            # Get value from trace (natural space) using .isel to prevent dimension errors
            val_natural = posterior[rv_name].isel(sample=idx).values

            value_var = model.rvs_to_values[rv]
            value_name = value_var.name
            transform = getattr(rv, 'transform', None)

            if transform is not None:
                try:
                    # Manually push the value through the PyTensor transform
                    val_transformed = transform.forward(pt.as_tensor(val_natural)).eval()
                    point[value_name] = val_transformed
                except Exception:
                    # If transform fails, skip to avoid breaking the whole loop
                    pass
            else:
                point[value_name] = val_natural
        # ------------------------------------------------

        # Evaluate the point dictionary
        try:
            ll = loglike_func(point)
            if np.isfinite(ll):
                log_likelihoods.append(ll)
        except Exception:
            pass

    ll_array = np.array(log_likelihoods)

    if len(ll_array) == 0:
        print("Failed to compute valid log-likelihoods.")
        return None

    # Compute Median Log-Likelihood
    median_log_likelihood = np.median(ll_array)
    print(f"\n✓ Successfully evaluated {len(ll_array)} draws.")
    print(f"Median Log-Likelihood: {median_log_likelihood:.4f}")

    # Plot Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(ll_array, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(median_log_likelihood, color='red', linestyle='dashed', linewidth=2,
                label=f'Median: {median_log_likelihood:.2f}')
    plt.title('Distribution of Log-Likelihoods Across MCMC Draws')
    plt.xlabel('Log-Likelihood')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()

    plot_filename = 'Gillon_log_likelihood_hist.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"✓ Saved histogram to '{plot_filename}'")

    # Compute AIC and BIC
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
        'median_log_likelihood': median_log_likelihood,
        'll_std': np.std(ll_array)
    }

def get_dic_score(model, trace, samples=2000):
    """
    Computes DIC using model.compile_logp().
    Fixes 'nan' results and Celerite crashes by avoiding graph splicing.
    """
    import numpy as np
    from tqdm import tqdm

    print(f"--- Computing DIC (Samples: {samples}) ---")

    # 1. Compile Function
    with model:
        logp_func = model.compile_logp()

    # 2. Prepare Data Indices
    posterior = trace.posterior.stack(sample=("chain", "draw"))
    n_total = posterior.sizes["sample"]

    if n_total > samples:
        indices = np.random.choice(n_total, size=samples, replace=False)
    else:
        indices = np.arange(n_total)

    # 3. Prepare Base Point (Handles missing vars like u_star)
    base_point = model.initial_point()
    trace_vars = list(trace.posterior.data_vars)

    # Pre-fetch trace data to speed up the loop
    # We only fetch vars that exist in the model's point dictionary
    relevant_vars = [v for v in trace_vars if v in base_point]
    data_cache = {v: posterior[v].values for v in relevant_vars}

    # 4. Compute D_bar (Mean Deviance)
    deviances = []

    for i in tqdm(indices, desc="Calculating Deviance"):
        # Create a point for this sample
        point = base_point.copy()
        for var in relevant_vars:
            point[var] = data_cache[var][..., i]

        try:
            val = logp_func(point)
            deviances.append(-2 * val)
        except Exception:
            # Skip numerical errors
            pass

    d_bar = np.mean(deviances)

    # 5. Compute D_hat (Deviance at Median)
    median_point = base_point.copy()
    for var in relevant_vars:
        # Calculate median over the 'sample' dimension (last axis)
        median_point[var] = np.median(data_cache[var], axis=-1)

    d_hat = -2 * logp_func(median_point)

    # 6. Results
    p_d = d_bar - d_hat
    dic = d_bar + p_d

    print(f"Mean Deviance (D_bar):     {d_bar:.4f}")
    print(f"Effective Params (p_D):    {p_d:.4f}")
    print(f"DIC:                       {dic:.4f}")
    print("="*40 + "\n")

    return dic, p_d

def run_analysis(model, trace, map_sol, x, y, x_rv, y_rv, yerr_rv, id_rv, inst_names):
    print("\n--- Running Analysis ---")

    print("Saving RV GP results...")
    trace.to_netcdf("rv_lc_gp_Gillon_trace.nc")
    with open("rv_lc_gp_Gillon_map.pkl", "wb") as f:
        pickle.dump(map_sol, f)

    summ = az.summary(trace, var_names=["period", "m_pl", "ecc", "rv_offsets"])
    print(summ)

# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    x, y, yerr, x_rv, y_rv, yerr_rv, id_rv, inst_names, rv_medians, shk_x, shk_y_std = load_data()
    print(f"Loaded {len(x)} LC points, {len(x_rv)} RV points from {inst_names}, SHK: {len(shk_x)}")

    model = build_model(x, y, yerr, x_rv, y_rv, yerr_rv, id_rv, inst_names, rv_medians, shk_x, shk_y_std)

    with model:
        print("\n=== MAP Optimization (Iterative) ===")
        start_pt = model.initial_point()

        # Step 1: Lock phases. Only optimize Period and T0.
        print("1. Optimizing Phase (Period, T0)...")
        map_sol = pmx.optimize(start=start_pt, vars=[model['t0'], model['log_period']])

        # Step 2: Lock Baseline. Optimize Offsets, Jitters, Mean Flux, Activity & GP.
        print("2. Optimizing Baseline (Offsets, Jitters, Mean Flux, Activity, GP)...")
        vars_step2 = [
            model['rv_offsets'], model['mean_flux'], model['log_jitter_lc'], model['rv_jitters'],
            model['C_activity'], model['log_sigma_rv_gp'], model['log_rho_rv_gp']
        ]
        map_sol = pmx.optimize(start=map_sol, vars=vars_step2)

        print("2.5. Optimizing Light Curve GP (Rotation Only)...")
        vars_lc_gp = [
            model["log_P_rot"], model["log_Q0_lc"], model["log_dQ_lc"],
            model["f_rot_lc"], model["log_sigma_lc"]]
        map_sol = pmx.optimize(start=map_sol, vars=vars_lc_gp)

        # Step 3: Transit Geometry.
        # Optimize b, depth, r_star.
        print("3. Optimizing Geometry (b, depth, r_star)...")
        vars_step3 = [model['b_transiting'], model['log_depth'], model['r_star']]
        map_sol = pmx.optimize(start=map_sol, vars=vars_step3)

        # Step 4: Orbital Dynamics.
        # Now we fit the RV shape (ecc, omega, mass).
        print("4. Optimizing Orbits (Ecc, Omega, Mass)...")
        vars_step4 = [model['ecc'], model['omega'], model['log_m_pl']]
        map_sol = pmx.optimize(start=map_sol, vars=vars_step4)

        # Step 5: Full Polish
        # Release u_star now that geometry is stable
        print("5. Optimizing Full Model...")
        map_sol = pmx.optimize(start=map_sol)

        print("\n=== Checking Initial Gradients ===")
        check_gradient(model, map_sol)
        print_map_values(map_sol, inst_names)

        # Removed Pre-Sampling AIC/BIC as requested.

        print("\n=== Sampling ===")
        trace = pm.sample(
            tune=1500,
            draws=1500,
            initvals=map_sol,
            init="adapt_full",
            chains=4,
            cores=4,
            target_accept=0.98,
            return_inferencedata=True,
        )

    # Run Analysis
    run_analysis(model, trace, map_sol, x, y, x_rv, y_rv, yerr_rv, id_rv, inst_names)

    # Run Post-Sampling Metrics
    print("\n" + "="*60)
    print("COMPUTING MODEL COMPARISON METRICS")
    print("="*60)

    # Count total observations (Light Curve + RV)
    n_obs = len(x) + len(x_rv)
    print(f"Total Observations: {len(x)} (LC) + {len(x_rv)} (RV) = {n_obs}")

    # Compute DIC
    get_dic_score(model, trace)

    # Compute AIC/BIC (with corrected log-likelihood)
    results = get_aic_bic_metric(model, trace, n_obs)

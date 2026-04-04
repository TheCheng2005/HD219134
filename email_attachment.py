import numpy as np
import pandas as pd
import arviz as az
import celerite2
from celerite2 import terms
import exoplanet as xo
import astropy.units as u

# Import your model dependencies
from RV_LC_GP_4planet import load_data

print("--- Starting Best Draw & Residuals Extraction Script ---")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
RUN_NAME = "rv_lc_gp_4planet"
TRACE_FILE = f"{RUN_NAME}_trace.nc"

nplanet = 4
ntransit = 2
x_ref = 2450000.0  # Reference time to add back for full BJD

# =============================================================================
# 2. LOAD DATA & TRACE
# =============================================================================
print("Loading raw data from model script...")
x, y, yerr, x_rv, y_rv, yerr_rv, id_rv, inst_names, rv_medians, shk_rv_std = load_data()

# Replicate SHK mapping exactly as done in the model
shk_at_rv_val = shk_rv_std

print(f"Loading trace: {TRACE_FILE}")
trace = az.from_netcdf(TRACE_FILE)

# =============================================================================
# 3. IDENTIFY THE BEST DRAW & COMPUTE STD DEVIATIONS
# =============================================================================
print("Finding the best draw based on log-posterior (lp)...")
stacked_stats = trace.sample_stats.stack(sample=("chain", "draw"))
best_idx = int(stacked_stats.lp.argmax().values)

best_chain = int(stacked_stats.chain[best_idx].values)
best_draw = int(stacked_stats.draw[best_idx].values)
best_lp = float(stacked_stats.lp[best_idx].values)

print(f"Best Sample Found -> Chain: {best_chain}, Draw: {best_draw}, Log-Prob: {best_lp}")

stacked_post = trace.posterior.stack(sample=("chain", "draw"))
best_pars = {}
std_pars = {}

for var in stacked_post.data_vars:
    # Extract the exact MAP value for this specific draw
    best_pars[var] = stacked_post[var].isel(sample=best_idx).values
    # Compute the standard deviation across the entire posterior for this parameter
    std_pars[var] = stacked_post[var].std(dim="sample").values

# =============================================================================
# 4. EXPORT 1ST CSV: BEST PARAMETERS + STD DEV
# =============================================================================
print("Exporting best parameters and standard deviations to CSV...")
flat_pars = {}
flat_stds = {}

for key in best_pars.keys():
    val = np.atleast_1d(best_pars[key])
    std_val = np.atleast_1d(std_pars[key])

    if len(val) == 1:
        flat_pars[key] = val[0]
        flat_stds[key] = std_val[0]
    else:
        for i in range(len(val)):
            flat_pars[f"{key}_{i}"] = val[i]
            flat_stds[f"{key}_{i}"] = std_val[i]

df_params = pd.DataFrame({
    'Parameter': list(flat_pars.keys()),
    'Best_MAP_Value': list(flat_pars.values()),
    'Std_Dev': list(flat_stds.values())
})

params_filename = f"{RUN_NAME}_best_draw_parameters.csv"
df_params.to_csv(params_filename, index=False, float_format='%.10g')
print(f"-> Saved {params_filename}")

# =============================================================================
# 5. RECONSTRUCT MODELS (No Interpolation)
# =============================================================================
print("Reconstructing models and GPs strictly at observed timestamps...")

p = best_pars['period']
t0 = best_pars['t0']
ecc = best_pars['ecc']
omega = best_pars['omega']
m_pl = best_pars['m_pl']

if 'b_all' in best_pars:
    b = best_pars['b_all']
elif 'b_transiting' in best_pars:
    b_trans = best_pars['b_transiting']
    b_non = np.zeros(nplanet - len(b_trans))
    b = np.concatenate([b_trans, b_non])
else:
    b = np.zeros(nplanet)

# LC Keplerian
u_star = best_pars.get('u_star', [0.426, 0.207])
star = xo.LimbDarkLightCurve(u_star)
r_pl = best_pars['r_pl']

orbit_lc = xo.orbits.KeplerianOrbit(
    r_star=best_pars['r_star'], m_star=best_pars['m_star'],
    period=p[:ntransit], t0=t0[:ntransit], b=b[:ntransit],
    m_planet=xo.units.with_unit(m_pl[:ntransit], u.Mearth),
    ecc=ecc[:ntransit], omega=omega[:ntransit]
)
lc_model = star.get_light_curve(orbit=orbit_lc, r=r_pl, t=x, texp=120./86400).eval() * 1e3
total_lc_keplerian = np.sum(lc_model, axis=1) + float(best_pars['mean_flux'])

# RV Keplerian
orbit_rv = xo.orbits.KeplerianOrbit(
    r_star=best_pars['r_star'], m_star=best_pars['m_star'],
    period=p, t0=t0, b=b, m_planet=xo.units.with_unit(m_pl, u.Mearth),
    ecc=ecc, omega=omega
)
rv_model = orbit_rv.get_radial_velocity(x_rv).eval()
total_rv_keplerian = np.sum(rv_model, axis=1)

# LC GP
sigma_lc = np.exp(best_pars['log_sigma_lc'])
P_rot = np.exp(best_pars['log_P_rot'])
Q0_lc = np.exp(best_pars['log_Q0_lc'])
dQ_lc = np.exp(best_pars['log_dQ_lc'])
f_rot_lc = best_pars['f_rot_lc']

kernel_lc = terms.RotationTerm(sigma=sigma_lc, period=P_rot, Q0=Q0_lc, dQ=dQ_lc, f=f_rot_lc)
gp_lc = celerite2.GaussianProcess(kernel_lc, mean=0.0)

inflated_yerr = np.sqrt(yerr**2 + np.exp(2 * best_pars['log_jitter_lc']))
gp_lc.compute(x, yerr=inflated_yerr)
mu_gp_lc = gp_lc.predict(y - total_lc_keplerian, t=x)

# RV GP & Activity
activity_model = best_pars['C_activity'] * shk_at_rv_val
total_rv_mean = total_rv_keplerian + best_pars['rv_offsets'][id_rv] + activity_model

sigma_rv = np.exp(best_pars['log_sigma_rv_gp'])
Q0_rv = np.exp(best_pars['log_Q0_rv'])
dQ_rv = np.exp(best_pars['log_dQ_rv'])
f_rot_rv = best_pars['f_rot_rv']
kernel_rv = terms.RotationTerm(sigma=sigma_rv, period=P_rot, Q0=Q0_rv, dQ=dQ_rv, f=f_rot_rv)

gp_rv = celerite2.GaussianProcess(kernel_rv, mean=0.0)

inflated_yerr_rv = np.sqrt(yerr_rv**2 + best_pars['rv_jitters'][id_rv]**2 + 1e-4)
gp_rv.compute(x_rv, yerr=inflated_yerr_rv)
mu_gp_rv = gp_rv.predict(y_rv - total_rv_mean, t=x_rv)

# =============================================================================
# 6. EXPORT 2ND SET OF CSVS: RESIDUALS (WITH BJD)
# =============================================================================
print("Exporting residuals to CSVs...")

# LC Residuals
lc_residuals = y - total_lc_keplerian - mu_gp_lc
df_lc = pd.DataFrame({
    'time_BJD': x + x_ref,
    'residual': lc_residuals,
    'e_residual': inflated_yerr
})
lc_res_filename = f"{RUN_NAME}_lc_residuals_best_draw.csv"
df_lc.to_csv(lc_res_filename, index=False, float_format='%.8f')
print(f"-> Saved {lc_res_filename}")

# RV Residuals
rv_residuals = y_rv - total_rv_mean - mu_gp_rv
rv_inst_labels = [inst_names[i] for i in id_rv]
df_rv = pd.DataFrame({
    'time_BJD': x_rv + x_ref,
    'residual': rv_residuals,
    'e_residual': inflated_yerr_rv,
    'instrument': rv_inst_labels
})
rv_res_filename = f"{RUN_NAME}_rv_residuals_best_draw.csv"
df_rv.to_csv(rv_res_filename, index=False, float_format='%.8f')
print(f"-> Saved {rv_res_filename}")

print("\nAll tasks completed successfully!")

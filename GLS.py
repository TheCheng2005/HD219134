import numpy as np
import matplotlib.pyplot as plt
import os
import arviz as az
import pytensor.tensor as pt
from astropy.timeseries import LombScargle
import exoplanet as xo
import astropy.units as u

def load_data():
    x_ref = 2450000.
    instruments = [
        ("HARPS",      "HD219134.HARPS.SHK.rv",         0),
        ("HJST",       "HD219134.HJST_Phase3.rv",       1),
        ("HIRES-post", "HD219134.HARADA_HIRES-post.rv", 2),
        ("Levy",       "HD219134.HARADA_Levy.rv",       3),
        ("Hamilton",   "HD219134.HARADA_Hamilton.rv",   4)
    ]

    rv_x_list, rv_y_list, rv_yerr_list, rv_id_list = [], [], [], []
    shk_x, shk_y = [], []

    for name, fname, inst_id in instruments:
        if os.path.exists(fname):
            data = np.loadtxt(fname, comments='#')
            t_inst = data[:, 0] - x_ref
            y_inst = data[:, 1]
            yerr_inst = data[:, 2]

            mask = np.isfinite(t_inst) & np.isfinite(y_inst) & np.isfinite(yerr_inst)
            rv_x_list.append(t_inst[mask])
            rv_y_list.append(y_inst[mask])
            rv_yerr_list.append(yerr_inst[mask])
            rv_id_list.append(np.full(np.sum(mask), inst_id, dtype=int))

            # Load SHK data if available
            if name == "HARPS" and data.shape[1] >= 4:
                s_val = data[:, 3]
                mask_s = np.isfinite(s_val) & np.isfinite(t_inst)
                shk_x = t_inst[mask_s]
                shk_y = s_val[mask_s]

    x_rv = np.concatenate(rv_x_list)
    y_rv = np.concatenate(rv_y_list)
    yerr_rv = np.concatenate(rv_yerr_list)
    id_rv = np.concatenate(rv_id_list)

    # Sort chronologically
    sort_rv = np.argsort(x_rv)
    x_rv, y_rv, yerr_rv, id_rv = x_rv[sort_rv], y_rv[sort_rv], yerr_rv[sort_rv], id_rv[sort_rv]

    # Standardize SHK exactly as done in PyMC model
    if len(shk_x) == 0:
        shk_x = x_rv
        shk_y = np.zeros_like(x_rv)
    idx_s = np.argsort(shk_x)
    shk_x, shk_y = shk_x[idx_s], shk_y[idx_s]
    shk_y_std = (shk_y - np.mean(shk_y)) / (np.std(shk_y) if np.std(shk_y) > 0 else 1.0)

    return x_rv, y_rv, yerr_rv, id_rv, shk_x, shk_y_std

def run_pre_whitening():
    trace_file = 'rv_lc_gp_4planet_trace.nc'
    if not os.path.exists(trace_file):
        print(f"Error: {trace_file} not found. Please run this in the directory with your trace.")
        return

    print("Loading raw RV data...")
    x_rv, y_rv, yerr_rv, id_rv, shk_x, shk_y_std = load_data()

    # Mirroring the SHK mapping from your PyMC model to correctly evaluate C_activity
    shk_at_rv = np.interp(x_rv, shk_x, shk_y_std)

    print(f"Loading trace: {trace_file}")
    trace = az.from_netcdf(trace_file)
    post = trace.posterior

    def get_med(var_name):
        return np.median(post[var_name].values, axis=(0, 1))

    print("Extracting median posterior parameters...")
    period = get_med('period')
    t0 = get_med('t0')
    m_pl = get_med('m_pl')
    ecc = get_med('ecc')
    omega = get_med('omega')
    m_star = get_med('m_star')
    r_star = get_med('r_star')
    b_all = get_med('b_all')
    rv_offsets = get_med('rv_offsets')

    try:
        C_activity = get_med('C_activity')
    except KeyError:
        C_activity = 0.0

    print(f"Planets being subtracted (Periods): {np.round(period, 2)} days")

    # Reconstruct the exact Keplerian model
    print("Computing Best-Fit RV Model...")
    orbit = xo.orbits.KeplerianOrbit(
        r_star=r_star, m_star=m_star,
        period=period, t0=t0, b=b_all,
        m_planet=xo.units.with_unit(m_pl, u.Mearth),
        ecc=ecc, omega=omega
    )

    # orbit.get_radial_velocity returns a PyTensor object. We sum over planets and evaluate it.
    rv_tensor = pt.sum(orbit.get_radial_velocity(x_rv), axis=-1)
    rv_orbits_eval = rv_tensor.eval()

    # Reconstruct noise and offset models
    activity_eval = C_activity * shk_at_rv
    offsets_eval = rv_offsets[id_rv]

    # Subtract the full structural model
    full_model = rv_orbits_eval + offsets_eval + activity_eval
    residuals = y_rv - full_model

    # Save residuals so you don't have to compute them again
    np.savetxt("rv_residuals_4planet.csv", np.column_stack([x_rv, residuals, yerr_rv]),
               delimiter=",", header="time,residual,yerr", comments='')
    print("Residuals saved to rv_residuals_4planet.csv")

    # Run Lomb-Scargle Periodogram on the Residuals
    print("Running Lomb-Scargle Periodogram on Residuals...")
    periods = np.logspace(np.log10(1.1), np.log10(3000), 10000)
    freqs = 1.0 / periods
    ls = LombScargle(x_rv, residuals, yerr_rv)
    power = ls.power(freqs)
    fap_levels = ls.false_alarm_level([0.1, 0.01, 0.001])

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(periods, power, color='black', lw=1.5)

    lit_planets = [3.09, 6.76, 22.72, 46.73, 94.26, 192.0, 2244.0]
    colors = ['r', 'r', 'b', 'g', 'm', 'orange', 'c']
    labels = ['b', 'c', '22d?', 'd', 'e', '192d?', 'h']

    for p, c, l in zip(lit_planets, colors, labels):
        plt.axvline(p, color=c, linestyle='--', alpha=0.7, label=f'{l} ({p}d)')

    plt.axhline(fap_levels[1], color='grey', linestyle=':', label='1% FAP')
    plt.axhline(fap_levels[2], color='black', linestyle=':', label='0.1% FAP')
    plt.xscale('log')
    plt.xlabel('Period (days)')
    plt.ylabel('Lomb-Scargle Power')
    plt.title('Residual Periodogram (4-Planet Orbits + Offsets + Activity Removed)')
    plt.legend(loc='upper right', fontsize=9, ncol=2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_pre_whitening()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import AutoMinorLocator
from scipy import stats
import arviz as az
import corner
import astropy.units as u
import pickle
import sys
import os
import exoplanet as xo
import celerite2
from celerite2 import terms

# === FIXED IMPORT ===
from RV_LC_GP_4planet import build_model, load_data

print("--- Starting General Analysis Script ---")
# =============================================================================
# 1. CONFIGURATION
# =============================================================================
RUN_NAME = "rv_lc_gp_4planet"

TRACE_FILE = f"{RUN_NAME}_trace.nc"
MAP_FILE   = f"{RUN_NAME}_map.pkl"

# Constants
x_ref    = 2450000.
nplanet  = 4
ntransit = 2
planet_names = ['b', 'c', 'd', 'h']

# =============================================================================
# 2. DATA LOADING (RELIANT ON MODEL SCRIPT)
# =============================================================================
print("Loading raw data from model script...")
# Updated to match the new return signature (no more shk_x, shk_y_std)
x, y, yerr, x_rv, y_rv, yerr_rv, id_rv, inst_names, rv_medians, shk_rv_std = load_data()

print(f"Data Loaded: {len(x)} LC points, {len(x_rv)} RV points.")

# =============================================================================
# 3. LOAD RESULTS & COMPUTE MEDIANS
# =============================================================================
try:
    print(f"Loading results for run: {RUN_NAME}")
    trace   = az.from_netcdf(TRACE_FILE)
    with open(MAP_FILE, "rb") as f:
        map_soln = pickle.load(f)

    print("Computing Posterior Medians...")
    best_pars = dict(map_soln)
    posterior = trace.posterior
    for var_name in posterior.data_vars:
        best_pars[var_name] = np.median(posterior[var_name].values, axis=(0, 1))

except FileNotFoundError:
    print(f"FATAL: Could not find {TRACE_FILE} or {MAP_FILE}.")
    sys.exit()

# =============================================================================
# 4. RECONSTRUCTION HELPERS
# =============================================================================
def get_model_components(params, t_lc, t_rv):
    """
    Reconstructs the Keplerian model arrays (LC and RV) from a parameter dictionary.
    """
    p     = params['period']
    t0    = params['t0']
    ecc   = params['ecc']
    omega = params['omega']
    m_pl  = params['m_pl']

    if 'b_all' in params:
        b = params['b_all']
    elif 'b_transiting' in params:
        b_trans = params['b_transiting']
        b_non   = np.zeros(nplanet - len(b_trans))
        b       = np.concatenate([b_trans, b_non])
    else:
        b = np.zeros(nplanet)

    # --- 1. LIGHT CURVE (Transiting Only) ---
    u_star = params.get('u_star', [0.426, 0.207])
    star   = xo.LimbDarkLightCurve(u_star)
    r_pl   = params['r_pl']

    orbit_lc = xo.orbits.KeplerianOrbit(
        r_star=params['r_star'],
        m_star=params['m_star'],
        period=p[:ntransit],
        t0=t0[:ntransit],
        b=b[:ntransit],
        m_planet=xo.units.with_unit(m_pl[:ntransit], u.Mearth),
        ecc=ecc[:ntransit],
        omega=omega[:ntransit]
    )

    lc_model = star.get_light_curve(
        orbit=orbit_lc, r=r_pl, t=t_lc, texp=120./86400
    ).eval() * 1e3

    # --- 2. RV (All 4 Planets) ---
    orbit_rv = xo.orbits.KeplerianOrbit(
        r_star=params['r_star'],
        m_star=params['m_star'],
        period=p, t0=t0, b=b,
        m_planet=xo.units.with_unit(m_pl, u.Mearth),
        ecc=ecc, omega=omega
    )

    rv_model = orbit_rv.get_radial_velocity(t_rv).eval()

    return lc_model, rv_model


def get_baselines(params, lc_comp_best, rv_comp_best):
    """
    Reconstructs the GP predictive means and Activity model strictly at observed timestamps.
    """
    # Shared rotation period
    P_rot = np.exp(params['log_P_rot'])

    # 1. LC GP (Rotation)
    total_lc_model = np.sum(lc_comp_best, axis=1) + float(params['mean_flux'])

    sigma_lc = np.exp(params['log_sigma_lc'])
    Q0_lc    = np.exp(params['log_Q0_lc'])
    dQ_lc    = np.exp(params['log_dQ_lc'])
    f_rot_lc = params['f_rot_lc']

    kernel_lc = terms.RotationTerm(sigma=sigma_lc, period=P_rot, Q0=Q0_lc, dQ=dQ_lc, f=f_rot_lc)
    gp_lc     = celerite2.GaussianProcess(kernel_lc, mean=0.0)
    gp_lc.compute(x, yerr=np.sqrt(yerr**2 + np.exp(2*params['log_jitter_lc'])))

    mu_gp_lc = gp_lc.predict(y - total_lc_model, t=x)

    # 2. RV Activity & GP (Rotation)
    # Using strict shk_rv_std directly mapped without interpolation
    activity_model = params['C_activity'] * shk_rv_std
    total_rv_mean  = np.sum(rv_comp_best, axis=1) + params['rv_offsets'][id_rv] + activity_model

    sigma_rv = np.exp(params['log_sigma_rv_gp'])
    Q0_rv    = np.exp(params['log_Q0_rv'])
    dQ_rv    = np.exp(params['log_dQ_rv'])
    f_rot_rv = params['f_rot_rv']

    kernel_rv = terms.RotationTerm(sigma=sigma_rv, period=P_rot, Q0=Q0_rv, dQ=dQ_rv, f=f_rot_rv)

    gp_rv = celerite2.GaussianProcess(kernel_rv, mean=0.0)
    gp_rv.compute(x_rv, yerr=np.sqrt(yerr_rv**2 + params['rv_jitters'][id_rv]**2 + 1e-4))

    mu_gp_rv = gp_rv.predict(y_rv - total_rv_mean, t=x_rv)

    return mu_gp_lc, mu_gp_rv, activity_model


def phase_fold(time, period, tc):
    return ((time - tc) / period + 0.5) % 1.0 - 0.5


# =============================================================================
# 5. SHARED PLOT HELPERS  (matching co-author notebook style)
# =============================================================================

# Instrument appearance — extend palette if you have more than 14 instruments
_XKCD_COLORS = [
    'xkcd:tomato red', 'xkcd:bright orange', 'xkcd:golden yellow', 'xkcd:butter',
    'xkcd:bright green', 'xkcd:green', 'xkcd:jungle green', 'xkcd:aqua',
    'xkcd:bright cyan', 'xkcd:vivid blue', 'xkcd:indigo blue', 'xkcd:bright lilac',
    'xkcd:purple', 'xkcd:bright pink',
]
_MARKERS = ['.', 'o', '^', '8', 's', '>', 'p', '<', '*', 'h', 'X', 'D', 'v', '+']

def _inst_color(uid):
    return _XKCD_COLORS[int(uid) % len(_XKCD_COLORS)]

def _inst_marker(uid):
    return _MARKERS[int(uid) % len(_MARKERS)]

def _inst_label(uid):
    """Mathtext-formatted instrument label."""
    name = inst_names[uid] if uid < len(inst_names) else f'Inst {uid}'
    return name  # plain string; mathtext not needed for legend labels


def _phased_0to1(time, tc, period, values):
    """Phase-fold onto [0, 1) and sort by phase (matches co-author's phased())."""
    phase = ((time - tc) / period) - np.floor((time - tc) / period)
    sort  = np.argsort(phase)
    return phase[sort], values[sort]


def _weighted_bin(phase, rv, err, num_bins=20):
    """Inverse-variance-weighted binning (matches co-author's binning())."""
    weights = 1.0 / err**2
    sum_w,  _, _   = stats.binned_statistic(phase, weights,    statistic='sum', bins=num_bins)
    sum_wy, edges, _ = stats.binned_statistic(phase, rv*weights, statistic='sum', bins=num_bins)
    avg      = np.divide(sum_wy, sum_w,
                         out=np.full_like(sum_wy, np.nan), where=sum_w != 0)
    centers  = 0.5 * (edges[:-1] + edges[1:])
    bin_err  = np.where(sum_w > 0, 1.0 / np.sqrt(sum_w), np.nan)
    return centers, avg, bin_err


def _apply_shared_rcparams():
    """Apply the rcParams used by the co-author's notebook (no system LaTeX needed)."""
    plt.rcParams['text.usetex']       = False   # use matplotlib mathtext instead
    plt.rcParams['mathtext.fontset']  = 'stix'  # closest look to LaTeX Computer Modern
    plt.rcParams['font.family']       = 'STIXGeneral'
    plt.rcParams['xtick.direction']   = 'in'
    plt.rcParams['ytick.direction']   = 'in'
    plt.rcParams.update({'font.size': 18})
    plt.rcParams['savefig.dpi']       = 150


# =============================================================================
# 6. PLOTTING FUNCTIONS
# =============================================================================

def plot_lc_analysis(best_pars, trace):
    print("Plotting LC Analysis by Sector...")

    lc_comp_best, rv_comp_best = get_model_components(best_pars, x, x_rv)
    mu_gp_lc, _, _ = get_baselines(best_pars, lc_comp_best, rv_comp_best)

    mean_flux   = float(best_pars['mean_flux'])
    total_model = np.sum(lc_comp_best, axis=1) + mean_flux

    # Identify sectors by finding large time gaps (> 10 days) in the x array
    gaps         = np.where(np.diff(x) > 10)[0]
    sector_bounds = [0] + list(gaps + 1) + [len(x)]
    sector_names  = ['S17', 'S24', 'S57', 'S77', 'S84']

    # Generate one figure per transiting planet
    for i in range(ntransit):
        fig, axes = plt.subplots(len(sector_names), 1,
                                 figsize=(8, 3 * len(sector_names)), sharex=True)

        p  = best_pars['period'][i]
        t0 = best_pars['t0'][i]

        # Detrend: Remove other planets, mean flux, AND the GP baseline
        other_planets = total_model - lc_comp_best[:, i]
        resid         = y - other_planets - mu_gp_lc
        ph            = phase_fold(x, p, t0)

        for s_idx in range(len(sector_names)):
            ax    = axes[s_idx]
            start, end = sector_bounds[s_idx], sector_bounds[s_idx+1]

            ph_sec    = ph[start:end]
            resid_sec = resid[start:end]
            sort      = np.argsort(ph_sec)

            ax.plot(ph_sec[sort], resid_sec[sort], ".k",
                    alpha=0.4, ms=3, label=f"Data ({sector_names[s_idx]})")

            model_sec = lc_comp_best[start:end, i]
            ax.plot(ph_sec[sort], model_sec[sort], "r-",
                    lw=2, label="Posterior Median Model")

            ax.set_title(f"Planet {planet_names[i]} - TESS Sector {sector_names[s_idx]}")
            ax.set_ylabel("Flux [ppt]")
            ax.set_xlim(-0.04, 0.04)

            if s_idx == 0:
                ax.legend(loc='upper right', fontsize='small')

        axes[-1].set_xlabel("Phase")
        plt.tight_layout()
        plot_name = f"{RUN_NAME}_lc_analysis_planet_{planet_names[i]}.png"
        plt.savefig(plot_name, dpi=150)
        plt.close()
        print(f"Saved {plot_name}")


# ---------------------------------------------------------------------------
def plot_rv_analysis(best_pars):
    print("Plotting RV Analysis (co-author style)...")
    _apply_shared_rcparams()

    lc_comp_best, rv_comp_best = get_model_components(best_pars, x, x_rv)
    _, mu_gp_rv, activity_model = get_baselines(best_pars, lc_comp_best, rv_comp_best)

    offsets              = best_pars['rv_offsets']
    total_rv_keplerian   = np.sum(rv_comp_best, axis=1)   # at observed timestamps

    # Stellar baseline to remove: instrument offsets + SHK activity + GP red noise
    stellar_baseline = offsets[id_rv] + activity_model + mu_gp_rv

    # Offset-corrected data (all instruments aligned to zero systemic velocity)
    rv_aligned = y_rv - stellar_baseline

    # Full residuals (all Keplerians also removed)
    res_all = rv_aligned - total_rv_keplerian

    # Dense time/phase grid for smooth model curves
    t_dense = np.linspace(x_rv.min(), x_rv.max(), 5000)
    orb_dense = xo.orbits.KeplerianOrbit(
        r_star=best_pars['r_star'],
        m_star=best_pars['m_star'],
        period=best_pars['period'],
        t0=best_pars['t0'],
        m_planet=xo.units.with_unit(best_pars['m_pl'], u.Mearth),
        ecc=best_pars['ecc'],
        omega=best_pars['omega'],
    )
    rv_dense_all   = orb_dense.get_radial_velocity(t_dense).eval()  # (5000, nplanet)
    rv_dense_total = np.sum(rv_dense_all, axis=1)

    fig = plt.figure(figsize=(10, 5 + 4 * nplanet))

    outer_gs = GridSpec(2, 1,
                        height_ratios=[2, nplanet],
                        hspace=0.13)

    top_gs = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs[0],
        height_ratios=[10, 3], hspace=0.0)
    ax0 = fig.add_subplot(top_gs[0])
    ax1 = fig.add_subplot(top_gs[1], sharex=ax0)

    bottom_gs = GridSpecFromSubplotSpec(
        nplanet, 1, subplot_spec=outer_gs[1],
        height_ratios=[10] * nplanet, hspace=0.0)
    phase_axes = [fig.add_subplot(bottom_gs[i]) for i in range(nplanet)]
    for i in range(1, nplanet):
        phase_axes[i].sharex(phase_axes[0])

    # Narrow the phase panels to match notebook's half-width layout
    fig.canvas.draw()   # needed to get actual positions
    for ax in phase_axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width * 0.7, pos.height])

    props = dict(boxstyle='round', facecolor='lavender',
                 edgecolor='black', linewidth=1, alpha=0.65)

    for uid in np.unique(id_rv):
        mask  = id_rv == uid
        color = _inst_color(uid)
        mark  = _inst_marker(uid)
        lab   = _inst_label(uid)

        ax0.plot(x_rv[mask] - x_ref, rv_aligned[mask],
                 linestyle='None', marker=mark, c=color,
                 label=lab, alpha=0.50, zorder=0)
        ax0.errorbar(x_rv[mask] - x_ref, rv_aligned[mask],
                     yerr=yerr_rv[mask],
                     linestyle='None', marker=mark, c=color,
                     capsize=3, alpha=0.50, zorder=0)
        ax1.plot(x_rv[mask] - x_ref, res_all[mask],
                 linestyle='None', marker=mark, c=color, alpha=0.10)

    ax0.plot(t_dense - x_ref, rv_dense_total,
             color='blue', linewidth=2, alpha=0.5, label='Full model')
    ax0.axhline(0, linestyle='--', color='black')
    ax0.set_ylabel(r'$\mathrm{RV\ (m/s)}$', fontsize=20)
    ax0.tick_params('x', labelbottom=False)
    ax0.text(0.01, 0.94, r'$\mathbf{a)}$',
             transform=ax0.transAxes, fontsize=12)
    ax0.xaxis.set_minor_locator(AutoMinorLocator())
    ax0.yaxis.set_minor_locator(AutoMinorLocator())

    ax1.axhline(0, linestyle='--', color='black')
    ax1.set_ylabel(r'$\mathrm{Residual}$', fontsize=20)
    ax1.set_xlabel(r'$\mathrm{BJD_{TDB} - 2450000}$')
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())

    handles, labels_leg = ax0.get_legend_handles_labels()
    phase_axes[2].legend(handles, labels_leg,
                         bbox_to_anchor=(1.05, 1.0), loc="lower left",
                         borderaxespad=0., fontsize=12, frameon=True)

    for i, (ax, pname) in enumerate(zip(phase_axes, planet_names)):
        p    = best_pars['period'][i]
        t0   = best_pars['t0'][i]
        ecc_i = best_pars['ecc'][i]

        other_keplerian = total_rv_keplerian - rv_comp_best[:, i]
        y_planet_i      = rv_aligned - other_keplerian     # planet i signal only

        ph_data,  rv_data  = _phased_0to1(x_rv, t0, p, y_planet_i)
        _,        err_data = _phased_0to1(x_rv, t0, p, yerr_rv)
        _,        id_data  = _phased_0to1(x_rv, t0, p, id_rv.astype(float))
        id_data = id_data.astype(int)

        bin_ph, bin_rv, bin_err = _weighted_bin(ph_data, rv_data, err_data, num_bins=20)

        ph_model, rv_model_i = _phased_0to1(t_dense, t0, p, rv_dense_all[:, i])

        K_i = 0.5 * (rv_model_i.max() - rv_model_i.min())

        for uid in np.unique(id_data):
            mask  = id_data == uid
            color = _inst_color(uid)
            mark  = _inst_marker(uid)
            ax.plot(ph_data[mask], rv_data[mask],
                    linestyle='None', marker=mark, c=color,
                    alpha=0.10, zorder=0)
            ax.errorbar(ph_data[mask], rv_data[mask],
                        yerr=err_data[mask],
                        linestyle='None', marker=mark, c=color,
                        capsize=3, alpha=0.10, zorder=0)

        ax.plot(bin_ph, bin_rv, 'ko',
                markersize=7, markerfacecolor='none', zorder=2)
        ax.errorbar(bin_ph, bin_rv, yerr=bin_err,
                    fmt='o', markersize=7, markerfacecolor='none',
                    capsize=3, color='black', zorder=2)

        ax.plot(ph_model, rv_model_i, 'red', linewidth=3, zorder=1)

        ax.set_ylabel(r'$\mathrm{RV\ (m/s)}$', fontsize=20, labelpad=5)
        ax.set_xlim([0, 1.0])
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())

        if i < nplanet - 1:
            ax.tick_params('x', labelbottom=False)
        else:
            ax.set_xlabel(r'$\mathrm{Phase}$')

        p_str = f'{p:.0f}' if p > 100 else f'{p:.3f}'
        textstr = (
            f'$K_{{{pname}}} = {K_i:.2f}$ m/s'
            f'     $P_{{{pname}}} = {p_str}$ days'
            f'     $e_{{{pname}}} = {ecc_i:.3f}$'
        )
        ax.text(0.12, 0.10, textstr,
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', ha='left', bbox=props)

        ax.text(0.01, 0.90, f'$\\mathbf{{{pname})}}$',
                transform=ax.transAxes, fontsize=12)

    plt.savefig(f"{RUN_NAME}_rv_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {RUN_NAME}_rv_analysis.png")


def plot_corners(trace):
    print("Generating Corner Plots...")

    transiting_indices = [0, 1]

    for i in range(nplanet):
        try:
            vars_to_plot = ["period", "t0", "m_pl", "ecc"]

            if i in transiting_indices:
                vars_to_plot.insert(2, "r_pl")
                vars_to_plot.insert(3, "b_transiting")

            samples, labels_corner = [], []

            for v in vars_to_plot:
                if v in trace.posterior:
                    val = trace.posterior[v].values
                    if val.ndim == 3 and val.shape[2] > i:
                        samples.append(val[:, :, i].flatten())
                        labels_corner.append(v)

            if samples:
                data = np.column_stack(samples)
                fig  = corner.corner(data, labels=labels_corner,
                                     quantiles=[0.16, 0.5, 0.84], show_titles=True)
                fig.suptitle(f"Planet {planet_names[i]}")
                fig.savefig(f"{RUN_NAME}_corner_{planet_names[i]}.png", bbox_inches='tight')
                plt.close()
                print(f"Saved {RUN_NAME}_corner_{planet_names[i]}.png")
            else:
                print(f"No samples found to plot for {planet_names[i]}")

        except Exception as e:
            print(f"Skipping corner for {planet_names[i]}: {e}")

    try:
        print("Generating System & GP Corner Plot...")
        # Updated to include RotationTerm parameters instead of SHOTerm
        vars_system = [
            "m_star", "r_star", "C_activity",
            "log_sigma_rv_gp", "log_Q0_rv", "log_dQ_rv", "f_rot_rv",
            "log_P_rot", "log_Q0_lc", "log_dQ_lc", "f_rot_lc", "log_sigma_lc"
        ]
        samples_sys, labels_sys = [], []

        for v in vars_system:
            if v in trace.posterior:
                val = trace.posterior[v].values
                samples_sys.append(val.flatten())
                labels_sys.append(v)

        if samples_sys:
            data_sys = np.column_stack(samples_sys)
            fig = corner.corner(data_sys, labels=labels_sys,
                                quantiles=[0.16, 0.5, 0.84], show_titles=True)
            fig.suptitle("System & GP Parameters")
            fig.savefig(f"{RUN_NAME}_corner_system_gp.png", bbox_inches='tight')
            plt.close()
            print(f"Saved {RUN_NAME}_corner_system_gp.png")

    except Exception as e:
        print(f"Skipping system corner plot: {e}")


def plot_gp_timeseries(best_pars):
    print("Plotting Full Time-Series GP Showcase...")

    lc_comp_best, rv_comp_best = get_model_components(best_pars, x, x_rv)
    mu_gp_lc, mu_gp_rv, activity_model = get_baselines(best_pars, lc_comp_best, rv_comp_best)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    offsets              = best_pars['rv_offsets']
    total_keplerian_rv   = np.sum(rv_comp_best, axis=1)
    y_rv_aligned         = y_rv - offsets[id_rv]

    ax1.errorbar(x_rv, y_rv_aligned, yerr=yerr_rv,
                 fmt='.k', alpha=0.4, label='Aligned Data')

    baseline_rv = mu_gp_rv + activity_model
    ax1.plot(x_rv, baseline_rv,
             color='orange', lw=2, zorder=5, label='Stellar Baseline (GP + SHK)')

    full_rv_model = total_keplerian_rv + baseline_rv
    ax1.plot(x_rv, full_rv_model,
             color='red', lw=1.5, zorder=10, alpha=0.8,
             label='Full Model (Orbits + Baseline)')

    ax1.set_title("Radial Velocity: Time Series & GP Baseline")
    ax1.set_ylabel("RV [m/s] (Offset Removed)")
    ax1.set_xlabel("Time [BJD - 2450000]")
    ax1.legend(loc='upper right', fontsize='small', ncol=3)

    mean_flux           = float(best_pars['mean_flux'])
    total_lc_keplerian  = np.sum(lc_comp_best, axis=1)

    gap_idx       = np.where(np.diff(x) > 10)[0][0]
    x_s17         = x[:gap_idx]
    y_s17         = y[:gap_idx]
    mu_gp_lc_s17  = mu_gp_lc[:gap_idx]
    keplerian_s17 = total_lc_keplerian[:gap_idx]

    ax2.plot(x_s17, y_s17, '.k', alpha=0.2, ms=2, label='Raw TESS Flux')
    ax2.plot(x_s17, mu_gp_lc_s17 + mean_flux,
             color='orange', lw=2, zorder=5, label='GP Rotation Baseline')
    ax2.plot(x_s17, keplerian_s17 + mu_gp_lc_s17 + mean_flux,
             color='red', lw=1, zorder=10, label='Full Model')

    ax2.set_title("Light Curve (Sector 17 Focus): Stellar Rotation GP")
    ax2.set_ylabel("Flux [ppt]")
    ax2.set_xlabel("Time [BJD - 2450000]")
    ax2.legend(loc='upper right', fontsize='small', ncol=3)

    plt.tight_layout()
    plt.savefig(f"{RUN_NAME}_gp_timeseries_showcase.png", dpi=150)
    plt.close()
    print(f"Saved {RUN_NAME}_gp_timeseries_showcase.png")


def plot_residuals_timeseries(best_pars):
    print("Plotting Residuals Time-Series...")

    lc_comp_best, rv_comp_best = get_model_components(best_pars, x, x_rv)
    mu_gp_lc, mu_gp_rv, activity_model = get_baselines(best_pars, lc_comp_best, rv_comp_best)

    total_rv_model = np.sum(rv_comp_best, axis=1)
    offsets        = best_pars['rv_offsets']
    resid_rv = y_rv - total_rv_model - offsets[id_rv] - activity_model - mu_gp_rv

    mean_flux      = float(best_pars['mean_flux'])
    total_lc_model = np.sum(lc_comp_best, axis=1) + mean_flux
    resid_lc       = y - total_lc_model - mu_gp_lc

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    colors = ['k', 'b', 'g', 'm', 'c']
    for uid in np.unique(id_rv):
        mask = (id_rv == uid)
        if np.any(mask):
            ax1.errorbar(x_rv[mask], resid_rv[mask], yerr=yerr_rv[mask],
                         fmt='.', color=colors[uid % len(colors)], alpha=0.6,
                         label=inst_names[uid])

    ax1.axhline(0, color='red', lw=2, zorder=10, label='Zero Residual')
    ax1.set_title("Radial Velocity Residuals")
    ax1.set_ylabel("Residual RV [m/s]")
    ax1.legend(loc='upper right', fontsize='small', ncol=6)

    gap_idx = np.where(np.diff(x) > 10)[0][0]
    ax2.plot(x[:gap_idx], resid_lc[:gap_idx], '.k', alpha=0.1, ms=2, label='TESS Residuals')
    ax2.axhline(0, color='red', lw=2, zorder=10)

    ax2.set_title("Light Curve Residuals")
    ax2.set_ylabel("Residual Flux [ppt]")
    ax2.set_xlabel("Time [BJD - 2450000]")
    ax2.set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig(f"{RUN_NAME}_residuals.png", dpi=150)
    plt.close()
    print(f"Saved {RUN_NAME}_residuals.png")


# =============================================================================
# 7. EXECUTION
# =============================================================================
if __name__ == "__main__":

    print("\n--- Summary Statistics ---")
    # Updated to track RotationTerm parameters instead of SHOTerm
    summary_vars = [
        "period", "t0", "m_pl", "ecc", "b_transiting", "mean_flux", "rv_jitters",
        "C_activity", "log_sigma_rv_gp", "log_Q0_rv", "log_dQ_rv", "f_rot_rv", "ror",
        "log_P_rot", "log_Q0_lc", "log_dQ_lc", "f_rot_lc", "log_sigma_lc"
    ]

    existing_vars = [v for v in summary_vars if v in trace.posterior]
    summary = az.summary(trace, var_names=existing_vars)

    print("\n=== Gelman-Rubin Statistics (R-hat) ===")
    print(summary[["r_hat", "ess_bulk"]])
    summary.to_csv(f"{RUN_NAME}_summary.csv")
    print(f"\nFull summary saved to {RUN_NAME}_summary.csv")

    # Plots
    plot_lc_analysis(best_pars, trace)
    plot_rv_analysis(best_pars)
    plot_gp_timeseries(best_pars)
    plot_residuals_timeseries(best_pars)
    plot_corners(trace)

    print("\nAnalysis Complete.")

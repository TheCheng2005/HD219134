import pandas as pd
import numpy as np
import arviz as az
import os
import re

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
# CSV files for each planet
CSV_PATTERN = 'lit_data/HD_219134_{}_planet_params.csv'

# Define all your models here in the exact order you want them to appear.
# The FIRST model in this dictionary that contains a planet will be used as the
# baseline Reference for the Sigma Differences table on that planet's sheet.
#
# Each model supports two mutually exclusive source keys:
#   'file'     : path to an ArviZ .nc trace file  (loaded via load_user_results)
#   'csv_file' : path to an EXOFASTv2 median CSV   (loaded via load_csv_results)
USER_MODELS = {
    '4-Planet Model': {
        'file': 'rv_lc_gp_4planet_trace.nc',
        'planets': ['b', 'c', 'd', 'h']
    },
    #'ExoFAST': {'csv_file': 'HD219134.6.median.csv','planets': ['b', 'c', 'f', 'd', 'h']},
    '6-Planet Model': {
        'file': 'rv_lc_gp_6planet_trace.nc',
        'planets': ['b', 'c', 'f', 'd', 'g', 'h']
    },
    '5-Planet_with_i Model': {
        'file': 'rv_lc_gp_5planet_with_i_trace.nc',
        'planets': ['b', 'c', 'd', 'i', 'h']
    },
    'Gillon Model': {
        'file': 'rv_lc_gp_Gillon_trace.nc',
        'planets': ['b', 'c', 'f', 'd']
    },
    'Harada Model': {
        'file': 'rv_lc_gp_Harada_trace.nc',
        'planets': ['b', 'c', 'f', 'd', 'i', 'h']}
}

OUTPUT_FILE = 'Comparison_Report_Combined.xlsx'
R_SUN_TO_R_EARTH = 109.076
T0_OFFSET = 2450000.0

# Physical constants
G = 6.67430e-11
M_SUN = 1.98892e30
R_SUN = 6.957e8
M_EARTH = 5.972e24
AU = 1.495978707e11

# Unit conversion factors for EXOFASTv2 CSV outputs
M_JUP = 1.89813e27          # kg
R_JUP = 7.1492e7            # m
M_JUP_TO_EARTH = M_JUP / M_EARTH   # ~317.83  (M_Jup → M_Earth)
R_JUP_TO_EARTH = (R_JUP / R_SUN) * R_SUN_TO_R_EARTH  # ~11.21  (R_Jup → R_Earth)

# =============================================================================
# PARSE VALUE WITH UNCERTAINTY
# =============================================================================
def parse_value_uncertainty(val_str):
    if pd.isna(val_str) or val_str == '---' or val_str.strip() == '':
        return np.nan, np.nan

    val_str = str(val_str).strip()

    # Format: value±uncertainty
    match = re.match(r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)±([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', val_str)
    if match:
        return float(match.group(1)), float(match.group(2))

    # Format: value+upper-lower
    match = re.match(r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\-([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', val_str)
    if match:
        value = float(match.group(1))
        upper = float(match.group(2))
        lower = float(match.group(3))
        return value, (upper + lower) / 2.0

    try:
        return float(val_str), np.nan
    except:
        return np.nan, np.nan

# =============================================================================
# LOAD LITERATURE
# =============================================================================
def load_literature_transposed(csv_path):
    if not os.path.exists(csv_path):
        return None

    df_raw = pd.read_csv(csv_path)
    param_col = df_raw.columns[0]
    source_cols = df_raw.columns[1:]
    data = []

    for source in source_cols:
        source_data = {'Reference': source}
        for _, row in df_raw.iterrows():
            param_name = row[param_col]
            value, uncertainty = parse_value_uncertainty(row[source])
            source_data[param_name] = value
            source_data[f'{param_name}_err'] = uncertainty
        data.append(source_data)

    return pd.DataFrame(data)

# =============================================================================
# COMPUTE DERIVED PARAMETERS
# =============================================================================
def compute_semi_major_axis(period_days, m_star_msun):
    period_sec = period_days * 86400.0
    m_star_kg = m_star_msun * M_SUN
    a_cubed = G * m_star_kg * period_sec**2 / (4 * np.pi**2)
    return a_cubed**(1.0/3.0) / AU

def compute_inclination(b, r_star_rsun, a_au):
    r_star_au = r_star_rsun * R_SUN / AU
    cos_i = np.clip(b * r_star_au / a_au, -1.0, 1.0)
    return np.degrees(np.arccos(cos_i))

def compute_rv_semiamplitude(m_pl_mearth, period_days, ecc, m_star_msun, sin_i):
    period_sec = period_days * 86400.0
    m_pl_kg = m_pl_mearth * M_EARTH
    m_star_kg = m_star_msun * M_SUN
    term1 = (2 * np.pi * G / period_sec)**(1.0/3.0)
    term2 = m_pl_kg * sin_i
    term3 = (m_star_kg + m_pl_kg)**(2.0/3.0)
    term4 = np.sqrt(1 - ecc**2)
    return term1 * term2 / term3 / term4

def compute_msini(m_pl_mearth, sin_i):
    return m_pl_mearth * sin_i

def circular_mean(angles_deg):
    angles_rad = np.radians(angles_deg)
    sin_mean = np.mean(np.sin(angles_rad))
    cos_mean = np.mean(np.cos(angles_rad))
    return np.degrees(np.arctan2(sin_mean, cos_mean)) % 360

def circular_std(angles_deg):
    angles_rad = np.radians(angles_deg)
    R = np.sqrt(np.mean(np.sin(angles_rad))**2 + np.mean(np.cos(angles_rad))**2)
    if R < 1e-10: return 180.0
    return np.degrees(np.sqrt(-2 * np.log(R)))

# =============================================================================
# LOAD USER RESULTS  (from ArviZ .nc trace)
# =============================================================================
def load_user_results(trace_path, planet_list):
    if not os.path.exists(trace_path):
        return None

    try:
        trace = az.from_netcdf(trace_path)
    except Exception as e:
        print(f"  [ERROR] Could not load trace: {e}")
        return None

    def get_val_err(param, planet_idx=None, scale=1.0, circular=False):
        """Extracts 50th percentile as value, and asymmetric bounds from 16th and 84th percentiles."""
        if param not in trace.posterior:
            return np.nan, np.nan, np.nan

        vals = trace.posterior[param].values
        if planet_idx is not None:
            if vals.ndim == 3:
                if planet_idx < vals.shape[-1]:
                    planet_vals = vals[..., planet_idx]
                else:
                    return np.nan, np.nan, np.nan
            else:
                planet_vals = vals
        else:
            planet_vals = vals

        flat_vals = planet_vals.flatten()
        if circular:
            # For angles, falling back to specific directional statistics rather than simple percentiles
            angles_deg = np.degrees(flat_vals) % 360
            value = circular_mean(angles_deg) * scale
            uncertainty = circular_std(angles_deg) * scale
            return value, uncertainty, uncertainty
        else:
            pct_16, pct_50, pct_84 = np.percentile(flat_vals, [16, 50, 84])
            value = pct_50 * scale
            lower_err = (pct_50 - pct_16) * scale
            upper_err = (pct_84 - pct_50) * scale
            return value, lower_err, upper_err

    results = {}
    m_star_val, m_star_err_l, m_star_err_u = get_val_err('m_star')
    r_star_val, r_star_err_l, r_star_err_u = get_val_err('r_star')

    # Pre-compute mean errors for propagation routines
    m_star_err = (m_star_err_l + m_star_err_u) / 2.0
    r_star_err = (r_star_err_l + r_star_err_u) / 2.0

    for i, letter in enumerate(planet_list):
        p_res = {}
        p_res['Period'] = get_val_err('period', planet_idx=i)
        p_res['Mass'] = get_val_err('m_pl', planet_idx=i)
        p_res['Ecc'] = get_val_err('ecc', planet_idx=i)

        val_w, err_w_l, err_w_u = get_val_err('omega', planet_idx=i, circular=True)
        p_res['Omega'] = (val_w, err_w_l, err_w_u) if not np.isnan(val_w) else (np.nan, np.nan, np.nan)

        val_t0, err_t0_l, err_t0_u = get_val_err('t0', planet_idx=i)
        p_res['T0'] = (val_t0 + T0_OFFSET, err_t0_l, err_t0_u) if not np.isnan(val_t0) else (np.nan, np.nan, np.nan)

        p_res['Impact_b'] = get_val_err('b_all', planet_idx=i)

        # Transiting check (assuming first 2 planets are transiting)
        p_res['Radius'] = get_val_err('r_pl', planet_idx=i, scale=R_SUN_TO_R_EARTH) if i < 2 else (np.nan, np.nan, np.nan)

        p_res['Stellar Mass'] = (m_star_val, m_star_err_l, m_star_err_u)
        p_res['Stellar Radius'] = (r_star_val, r_star_err_l, r_star_err_u)

        # Extract values for computed Parameters
        period_val, period_err_l, period_err_u = p_res['Period']
        mass_val, mass_err_l, mass_err_u = p_res['Mass']
        ecc_val, ecc_err_l, ecc_err_u = p_res['Ecc']
        b_val, b_err_l, b_err_u = p_res['Impact_b']

        # Use the mean error margin for simple analytic error propagation steps
        period_err = (period_err_l + period_err_u) / 2.0
        mass_err = (mass_err_l + mass_err_u) / 2.0
        ecc_err = (ecc_err_l + ecc_err_u) / 2.0
        b_err = (b_err_l + b_err_u) / 2.0

        if pd.notna(period_val) and pd.notna(m_star_val):
            a_val = compute_semi_major_axis(period_val, m_star_val)
            rel_err_p = period_err / period_val if period_val != 0 else 0
            rel_err_m = m_star_err / m_star_val if m_star_val != 0 else 0
            a_err = a_val * np.sqrt((2/3 * rel_err_p)**2 + (1/3 * rel_err_m)**2)
            p_res['SMA'] = (a_val, a_err, a_err) # Returned symmetrically
        else:
            p_res['SMA'] = (np.nan, np.nan, np.nan)

        sma_val, sma_err_l, sma_err_u = p_res['SMA']
        sma_err = (sma_err_l + sma_err_u) / 2.0

        if i < 2 and pd.notna(b_val) and pd.notna(r_star_val) and pd.notna(sma_val) and b_val > 0:
            incl_val = compute_inclination(b_val, r_star_val, sma_val)
            r_star_au = r_star_val * R_SUN / AU

            x = np.clip(b_val * r_star_au / sma_val, -1.0, 1.0)

            delta_x_sq = ((r_star_au / sma_val) * b_err)**2 + ((b_val / sma_val) * (r_star_err * R_SUN / AU))**2 + ((-b_val * r_star_au / (sma_val**2)) * sma_err)**2
            delta_x = np.sqrt(delta_x_sq)

            incl_err = np.degrees(1.0 / np.sqrt(1.0 - x**2) * delta_x) if abs(x) < 1.0 else 90.0
            p_res['Inclination'] = (incl_val, incl_err, incl_err)
        else:
            p_res['Inclination'] = (np.nan, np.nan, np.nan)

        incl_val, incl_err_l, incl_err_u = p_res['Inclination']
        if pd.notna(mass_val) and pd.notna(period_val) and pd.notna(ecc_val) and pd.notna(m_star_val):
            sin_i = np.sin(np.radians(incl_val)) if pd.notna(incl_val) else 1.0
            K_val = compute_rv_semiamplitude(mass_val, period_val, ecc_val, m_star_val, sin_i)
            K_err = K_val * (mass_err / mass_val if mass_val != 0 else 0)
            p_res['K'] = (K_val, K_err, K_err)
        else:
            p_res['K'] = (np.nan, np.nan, np.nan)

        if pd.notna(mass_val) and pd.notna(incl_val):
            sin_i = np.sin(np.radians(incl_val))
            msini_val = compute_msini(mass_val, sin_i)
            msini_err = mass_err * sin_i
            p_res['Mass_sini'] = (msini_val, msini_err, msini_err)
        else:
            p_res['Mass_sini'] = (np.nan, np.nan, np.nan)

        results[letter] = p_res

    return results

# =============================================================================
# LOAD CSV RESULTS  (from EXOFASTv2 median/errorbar CSV)
# =============================================================================
def _parse_exofast_multiplier(mult_str):
    """
    Convert a LaTeX-style multiplier column (e.g. r'\times 10^{-7}') to float.
    Returns 1.0 if the column is empty or unrecognised.
    """
    s = str(mult_str).strip() if not pd.isna(mult_str) else ''
    if not s:
        return 1.0
    match = re.search(r'10\^\{?([-+]?\d+)\}?', s)
    return 10 ** int(match.group(1)) if match else 1.0


def load_csv_results(csv_path, planet_list):
    """
    Load an EXOFASTv2 median/errorbar CSV and return the same result-dict
    structure as load_user_results().

    CSV format (comment line is skipped):
        #parname, median value, upper errorbar, lower errorbar[, \\times 10^{N}]

    Parameters are named  <parname>_<index>  where <index> is the 0-based
    position in planet_list (e.g. Period_0 → planet_list[0], mp_1 → planet_list[1]).

    EXOFASTv2 unit conventions handled here:
        mp   [M_Jup]  → converted to M_Earth
        rp   [R_Jup]  → converted to R_Earth  (only for transiting planets, i < 2)
        tc   [BJD]    → used directly as T0  (full BJD, no offset applied)
        omegadeg      → used directly  (degrees)
        omegagr       → fallback, converted from radians to degrees
        e    absent   → assumed 0.0 (circular orbit)
    """
    if not os.path.exists(csv_path):
        print(f"  [FAILED]  Could not find {csv_path}")
        return None

    # --- Parse every data row into a dict keyed by parname ---
    # Each value is stored as (median, lower_err, upper_err) after applying
    # any multiplier that appears in the optional 5th column.
    rows = {}
    with open(csv_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split(',')]
            parname = parts[0]
            try:
                mult  = _parse_exofast_multiplier(parts[4]) if len(parts) >= 5 else 1.0
                val   = float(parts[1]) * mult
                upper = float(parts[2]) * mult
                lower = float(parts[3]) * mult
            except (ValueError, IndexError):
                continue
            # Store as (median, lower_err, upper_err) — consistent with get_val_err
            rows[parname] = (val, lower, upper)

    def get(parname):
        """Look up a parsed row; return (nan, nan, nan) if not present."""
        return rows.get(parname, (np.nan, np.nan, np.nan))

    def scale_tuple(t, factor):
        """Multiply all three elements of a (val, lo, hi) tuple by factor."""
        return tuple(x * factor if not np.isnan(x) else x for x in t)

    # --- Stellar parameters (always planet index 0 in EXOFASTv2 output) ---
    m_star_val, m_star_err_l, m_star_err_u = get('mstar_0')
    r_star_val, r_star_err_l, r_star_err_u = get('rstar_0')
    m_star_err = (m_star_err_l + m_star_err_u) / 2.0
    r_star_err = (r_star_err_l + r_star_err_u) / 2.0

    results = {}
    for i, letter in enumerate(planet_list):
        p_res = {}

        # --- Period [days] ---
        p_res['Period'] = get(f'Period_{i}')

        # --- Mass: CSV in M_Jupiter → convert to M_Earth ---
        p_res['Mass'] = scale_tuple(get(f'mp_{i}'), M_JUP_TO_EARTH)

        # --- Radius: CSV in R_Jupiter → convert to R_Earth (transiting only) ---
        p_res['Radius'] = scale_tuple(get(f'rp_{i}'), R_JUP_TO_EARTH) if i < 2 else (np.nan, np.nan, np.nan)

        # --- Eccentricity (absent → 0.0 for a circular-orbit fit) ---
        ecc_raw = get(f'e_{i}')
        p_res['Ecc'] = ecc_raw if not np.isnan(ecc_raw[0]) else (0.0, 0.0, 0.0)

        # --- Argument of periastron: prefer degrees column, fall back to radians ---
        omega_deg = get(f'omegadeg_{i}')
        if np.isnan(omega_deg[0]):
            omega_gr = get(f'omegagr_{i}')
            omega_deg = scale_tuple(omega_gr, np.degrees(1.0)) if not np.isnan(omega_gr[0]) else (np.nan, np.nan, np.nan)
        p_res['Omega'] = omega_deg

        # --- T0: use tc (time of conjunction) which EXOFASTv2 reports in full BJD ---
        p_res['T0'] = get(f'tc_{i}')

        # --- Impact parameter ---
        p_res['Impact_b'] = get(f'b_{i}')

        # --- Stellar parameters (same for every planet) ---
        p_res['Stellar Mass']   = (m_star_val, m_star_err_l, m_star_err_u)
        p_res['Stellar Radius'] = (r_star_val, r_star_err_l, r_star_err_u)

        # --- Derived parameters (identical logic to load_user_results) ---
        period_val, period_err_l, period_err_u = p_res['Period']
        mass_val,   mass_err_l,   mass_err_u   = p_res['Mass']
        ecc_val,    ecc_err_l,    ecc_err_u    = p_res['Ecc']
        b_val,      b_err_l,      b_err_u      = p_res['Impact_b']

        period_err = (period_err_l + period_err_u) / 2.0
        mass_err   = (mass_err_l   + mass_err_u)   / 2.0
        ecc_err    = (ecc_err_l    + ecc_err_u)    / 2.0
        b_err      = (b_err_l      + b_err_u)      / 2.0

        if pd.notna(period_val) and pd.notna(m_star_val):
            a_val     = compute_semi_major_axis(period_val, m_star_val)
            rel_err_p = period_err / period_val if period_val != 0 else 0
            rel_err_m = m_star_err / m_star_val if m_star_val != 0 else 0
            a_err     = a_val * np.sqrt((2/3 * rel_err_p)**2 + (1/3 * rel_err_m)**2)
            p_res['SMA'] = (a_val, a_err, a_err)
        else:
            p_res['SMA'] = (np.nan, np.nan, np.nan)

        sma_val, sma_err_l, sma_err_u = p_res['SMA']
        sma_err = (sma_err_l + sma_err_u) / 2.0

        if i < 2 and pd.notna(b_val) and pd.notna(r_star_val) and pd.notna(sma_val) and b_val > 0:
            incl_val  = compute_inclination(b_val, r_star_val, sma_val)
            r_star_au = r_star_val * R_SUN / AU
            x = np.clip(b_val * r_star_au / sma_val, -1.0, 1.0)
            delta_x_sq = (
                    (r_star_au / sma_val * b_err)**2
                    + (b_val / sma_val * (r_star_err * R_SUN / AU))**2
                    + (-b_val * r_star_au / sma_val**2 * sma_err)**2
            )
            delta_x  = np.sqrt(delta_x_sq)
            incl_err = np.degrees(1.0 / np.sqrt(1.0 - x**2) * delta_x) if abs(x) < 1.0 else 90.0
            p_res['Inclination'] = (incl_val, incl_err, incl_err)
        else:
            p_res['Inclination'] = (np.nan, np.nan, np.nan)

        incl_val, incl_err_l, incl_err_u = p_res['Inclination']
        if pd.notna(mass_val) and pd.notna(period_val) and pd.notna(ecc_val) and pd.notna(m_star_val):
            sin_i = np.sin(np.radians(incl_val)) if pd.notna(incl_val) else 1.0
            K_val = compute_rv_semiamplitude(mass_val, period_val, ecc_val, m_star_val, sin_i)
            K_err = K_val * (mass_err / mass_val if mass_val != 0 else 0)
            p_res['K'] = (K_val, K_err, K_err)
        else:
            p_res['K'] = (np.nan, np.nan, np.nan)

        if pd.notna(mass_val) and pd.notna(incl_val):
            sin_i     = np.sin(np.radians(incl_val))
            msini_val = compute_msini(mass_val, sin_i)
            msini_err = mass_err * sin_i
            p_res['Mass_sini'] = (msini_val, msini_err, msini_err)
        else:
            p_res['Mass_sini'] = (np.nan, np.nan, np.nan)

        results[letter] = p_res

    return results

# =============================================================================
# GENERATE REPORT
# =============================================================================
def create_comparison_report(all_user_results, lit_data_dict, output_file):
    print(f"\nGenerating report: {output_file}...")

    param_map = [
        ('Period [days] (direct)', 'Period', 'P (days)'),
        ('Mass [M_earth] (direct)', 'Mass', 'Mp (M⨁)'),
        ('Radius [R_earth] (direct)', 'Radius', 'Rp (R⨁)'),
        ('Eccentricity (direct)', 'Ecc', 'e'),
        ('Omega [deg] (direct,circular)', 'Omega', 'ω (deg)'),
        ('Impact parameter b (direct)', 'Impact_b', 'b'),
        ('T0 [BJD] (direct+offset)', 'T0', 'TC (days)'),
        ('Semi-major axis [au] (computed)', 'SMA', 'a (au)'),
        ('Inclination [deg] (computed)', 'Inclination', 'i (deg)'),
        ('RV K [m/s] (computed)', 'K', 'K (m/s)'),
        ('Mass*sini [M_earth] (computed)', 'Mass_sini', 'Mp sin i (M⨁)'),
        ('Stellar Mass [M_sun]', 'Stellar Mass', None),
        ('Stellar Radius [R_sun]', 'Stellar Radius', None),
    ]

    all_possible_planets = sorted(list(set(sum([m['planets'] for m in USER_MODELS.values()], []))))

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for planet in all_possible_planets:
            df_lit = lit_data_dict.get(planet, pd.DataFrame())

            # --- FIND BASELINE REFERENCE FOR THIS PLANET ---
            # Finds the first model in USER_MODELS that successfully loaded and contains the planet.
            ref_model_name = None
            for m_name in USER_MODELS.keys():
                if m_name in all_user_results and planet in all_user_results[m_name]:
                    ref_model_name = m_name
                    break

            if not ref_model_name and df_lit.empty:
                continue # Skip sheet if no data exists at all for this planet

            ref_res = all_user_results.get(ref_model_name, {}).get(planet, {})

            # --- 1. BUILD VALUES TABLE ---
            data_values = []

            # User Models (Strict order)
            for m_name in USER_MODELS.keys():
                if m_name not in all_user_results or planet not in all_user_results[m_name]:
                    continue

                u_res = all_user_results[m_name][planet]
                row = {'Reference': m_name}
                for nice, user_key, _ in param_map:
                    res_tuple = u_res.get(user_key, (np.nan, np.nan, np.nan))
                    v = res_tuple[0]
                    if len(res_tuple) == 3:
                        e_low, e_high = res_tuple[1], res_tuple[2]
                    else:
                        e_low = e_high = res_tuple[1]

                    if pd.notna(v) and pd.notna(e_low) and pd.notna(e_high):
                        if abs(e_high - e_low) < 1e-6:
                            row[nice] = f"{v:.6f} ± {e_high:.6f}"
                        else:
                            row[nice] = f"{v:.6f} +{e_high:.6f} -{e_low:.6f}"
                    elif pd.notna(v):
                        row[nice] = f"{v:.6f}"
                    else:
                        row[nice] = "--"
                data_values.append(row)

            # Literature
            if not df_lit.empty:
                for _, lit in df_lit.iterrows():
                    row = {'Reference': lit['Reference']}
                    for nice, _, lit_col in param_map:
                        if lit_col is None:
                            row[nice] = "--"
                            continue
                        v, e = lit.get(lit_col, np.nan), lit.get(f'{lit_col}_err', np.nan)
                        if pd.notna(v) and pd.notna(e): row[nice] = f"{v:.6f} ± {e:.6f}"
                        elif pd.notna(v): row[nice] = f"{v:.6f}"
                        else: row[nice] = "--"
                    data_values.append(row)

            df_values = pd.DataFrame(data_values).set_index('Reference')
            sheet_name = f"Planet {planet}"
            df_values.to_excel(writer, sheet_name=sheet_name)

            # --- 2. BUILD SIGMA DIFFERENCES TABLE ---
            data_sig = []

            if ref_model_name:
                # Add reference row as pure zeros
                row_sig_ref = {'Reference': f"{ref_model_name} (Base)"}
                for nice, _, _ in param_map: row_sig_ref[nice] = 0.0
                data_sig.append(row_sig_ref)

                # Other User Models vs Base
                for m_name in USER_MODELS.keys():
                    if m_name == ref_model_name or m_name not in all_user_results or planet not in all_user_results[m_name]:
                        continue

                    u_res = all_user_results[m_name][planet]
                    row = {'Reference': m_name}
                    for nice, user_key, _ in param_map:
                        r_tuple = ref_res.get(user_key, (np.nan, np.nan, np.nan))
                        u_tuple = u_res.get(user_key, (np.nan, np.nan, np.nan))

                        r_v = r_tuple[0]
                        r_e = (r_tuple[1] + r_tuple[2]) / 2.0 if len(r_tuple) == 3 else r_tuple[1]

                        u_v = u_tuple[0]
                        u_e = (u_tuple[1] + u_tuple[2]) / 2.0 if len(u_tuple) == 3 else u_tuple[1]

                        if pd.notna(u_v) and pd.notna(r_v) and pd.notna(u_e) and pd.notna(r_e):
                            denom = np.sqrt(u_e**2 + r_e**2)
                            row[nice] = round(abs(u_v - r_v) / denom, 2) if denom > 0 else (0.0 if u_v == r_v else np.nan)
                        else:
                            row[nice] = np.nan
                    data_sig.append(row)

                # Literature vs Base
                if not df_lit.empty:
                    for _, lit in df_lit.iterrows():
                        row = {'Reference': lit['Reference']}
                        for nice, user_key, lit_col in param_map:
                            if lit_col is None:
                                row[nice] = np.nan
                                continue

                            r_tuple = ref_res.get(user_key, (np.nan, np.nan, np.nan))
                            r_v = r_tuple[0]
                            r_e = (r_tuple[1] + r_tuple[2]) / 2.0 if len(r_tuple) == 3 else r_tuple[1]

                            l_v, l_e = lit.get(lit_col, np.nan), lit.get(f'{lit_col}_err', np.nan)

                            if pd.notna(l_v) and pd.notna(r_v) and pd.notna(l_e) and pd.notna(r_e):
                                denom = np.sqrt(l_e**2 + r_e**2)
                                row[nice] = round(abs(l_v - r_v) / denom, 2) if denom > 0 else (0.0 if l_v == r_v else np.nan)
                            else:
                                row[nice] = np.nan
                        data_sig.append(row)

            df_sigma = pd.DataFrame(data_sig).set_index('Reference')
            worksheet = writer.sheets[sheet_name]

            # Formatting offsets based on the values dataframe size
            start_row_excel = len(df_values) + 4
            worksheet.cell(row=start_row_excel, column=1, value=f"SIGMA DIFFERENCES (vs {ref_model_name})")
            df_sigma.to_excel(writer, sheet_name=sheet_name, startrow=start_row_excel)

    print(f"✓ Report successfully saved to {output_file}")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("HD 219134 Multi-Model Literature Comparison")
    print("=" * 70)

    # 1. Load User Results safely
    all_user_results = {}
    print("\nLoading models:")
    for model_name, config in USER_MODELS.items():
        res = None

        if 'csv_file' in config:
            # ---- EXOFASTv2 median CSV ----
            res = load_csv_results(config['csv_file'], config['planets'])
            if res:
                print(f"  [SUCCESS] {model_name} loaded from {config['csv_file']}")

        elif 'file' in config:
            # ---- ArviZ trace (.nc) ----
            if os.path.exists(config['file']):
                res = load_user_results(config['file'], config['planets'])
                if res:
                    print(f"  [SUCCESS] {model_name} loaded from {config['file']}")
            else:
                print(f"  [FAILED]  Could not find {config['file']} for {model_name}")

        if res:
            all_user_results[model_name] = res

    if not all_user_results:
        print("\n[WARNING] No user trace files were loaded! Check your filenames.")

    # 2. Extract Literature target planets
    all_planets = sorted(list(set(sum([m['planets'] for m in USER_MODELS.values()], []))))

    # 3. Load Literature data
    lit_data_dict = {}
    print("\nLoading literature:")
    for planet in all_planets:
        csv_file = CSV_PATTERN.format(planet)
        df_lit = load_literature_transposed(csv_file)
        if df_lit is not None:
            lit_data_dict[planet] = df_lit

    if not lit_data_dict and not all_user_results:
        print("\n[ERROR] No valid trace files or literature data found. Exiting.")
        exit(1)

    create_comparison_report(all_user_results, lit_data_dict, OUTPUT_FILE)
    print("\n" + "=" * 70)
    print("Done!")

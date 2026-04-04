import numpy as np
import pandas as pd
import os

def analyze_transits_with_uncertainty(file_list, p, err_p, t0, err_t0):
    total_transits_all_files = 0

    print("--- TESS Light Curve Transit Analysis ---\n")

    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"[!] File not found: '{file_path}'. Skipping.")
            continue

        # 1. Load data strictly without interpolation
        try:
            # We only need the time column (index 0) to check bounds
            df = pd.read_csv(file_path, sep=r'\s+', comment='#', header=None, usecols=[0], names=['time'])
            time = df['time'].values
            # Filter out any NaNs in the raw data
            time = time[np.isfinite(time)]
        except Exception as e:
            print(f"[!] Error reading {file_path}: {e}")
            continue

        if len(time) == 0:
            print(f"File: {file_path} - No valid time data.")
            continue

        # 2. Calculate file duration
        t_min = np.min(time)
        t_max = np.max(time)
        duration = t_max - t_min

        # Determine possible epochs (E) covering this file's baseline
        e_min = int(np.floor((t_min - t0) / p))
        e_max = int(np.ceil((t_max - t0) / p))

        valid_transits_in_file = 0
        transit_details = []

        # 3. Check each epoch with rigorous error propagation
        for e in range(e_min, e_max + 1):
            tc = t0 + e * p

            # Error propagation formula
            err_tc = np.sqrt(err_t0**2 + (e * err_p)**2)

            # The entire uncertainty window must be within the file's baseline
            window_start = tc - err_tc
            window_end = tc + err_tc

            if window_start >= t_min and window_end <= t_max:
                valid_transits_in_file += 1
                transit_details.append((e, tc, err_tc))

        # Accumulate to global total
        total_transits_all_files += valid_transits_in_file

        # 4. Print the exact requested metrics per file
        print(f"File: {file_path}")
        print(f"  Duration:         {duration:.2f} days")
        print(f"  Valid Transits:   {valid_transits_in_file}")
        if valid_transits_in_file > 0:
            for e, tc, err_tc in transit_details:
                print(f"    -> Epoch {e:>4}: BJD {tc:.4f} ± {err_tc:.4f}")
        print("-" * 55)

    print(f"\nTOTAL VALIDATED TRANSITS ACROSS ALL FILES: {total_transits_all_files}")
    print("=======================================================\n")

if __name__ == "__main__":
    tess_files = [
        "n20191008.TESS.TESS.TIC283722336.S17.0120.SPOC.dat",
        "n20200416.TESS.TESS.TIC283722336.S24.0120.SPOC.dat",
        "n20220930.TESS.TESS.TIC283722336.S57.0120.SPOC.dat",
        "n20240325.TESS.TESS.TIC283722336.S77.0120.SPOC.dat",
        "n20241001.TESS.TESS.TIC283722336.S84.0120.SPOC.dat"
    ]

    # Planet parameters and uncertainties
    P = 94.257396
    err_P = 0.021608
    T0 = 2455983.075617
    err_T0 = 18.386592

    analyze_transits_with_uncertainty(tess_files, P, err_P, T0, err_T0)

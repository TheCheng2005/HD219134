"""
RV Dataset Overlap Checker — Ashley vs Chris
=============================================
Loads all RV files from two directories (Ashley's and Chris's),
computes BJD time coverage per file, and identifies overlapping datasets.

Produces:
  1. Console summary table of all overlapping file pairs
  2. Timeline (Gantt) plot — coverage per file, colour-coded by owner
  3. Overlay scatter plot — e_RV vs BJD for overlapping pairs only
  4. Overlay scatter plot — RV vs BJD for overlapping pairs only

Usage:
    python compare_rv_overlap.py [ashley_dir] [chris_dir]

    Defaults: ./Ashley   ./Chris
"""

import sys, os, glob, textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

ASHLEY_DIR = sys.argv[1] if len(sys.argv) > 1 else "./Ashley"
CHRIS_DIR  = sys.argv[2] if len(sys.argv) > 2 else "./Chris"
EXTENSIONS = ("*.rv.txt", "*.rv", "*.dat")
COMMENT    = "#"

# How close (in days) two BJD points must be to count as "same epoch"
EPOCH_MATCH_TOL = 0.01   # days — tighten/loosen as needed

# ── File I/O ──────────────────────────────────────────────────────────────────

def collect_files(directory):
    files = []
    for pat in EXTENSIONS:
        files.extend(glob.glob(os.path.join(directory, pat)))
    return sorted(set(files))


def short_label(filepath):
    name = Path(filepath).name
    for suf in (".rv.txt", ".rv", ".dat"):
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    return name


def load_rv(filepath):
    """Return (bjd, rv, e_rv) arrays or None."""
    try:
        data = np.loadtxt(filepath, comments=COMMENT)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        if data.shape[1] < 3:
            print(f"  [skip] {filepath} — fewer than 3 columns")
            return None
        return data[:, 0], data[:, 1], data[:, 2]
    except Exception as exc:
        print(f"  [skip] {filepath} — {exc}")
        return None

# ── Load both sets ─────────────────────────────────────────────────────────────

print(f"\nLoading Ashley  →  {ASHLEY_DIR}")
ashley_files = collect_files(ASHLEY_DIR)
print(f"Loading Chris   →  {CHRIS_DIR}\n")
chris_files  = collect_files(CHRIS_DIR)

if not ashley_files:
    sys.exit(f"No RV files found in Ashley dir: {ASHLEY_DIR}")
if not chris_files:
    sys.exit(f"No RV files found in Chris dir:  {CHRIS_DIR}")


def build_catalog(file_list, owner):
    catalog = []
    for fpath in file_list:
        result = load_rv(fpath)
        if result is None:
            continue
        bjd, rv, e_rv = result
        catalog.append({
            "owner":  owner,
            "label":  short_label(fpath),
            "path":   fpath,
            "bjd":    bjd,
            "rv":     rv,
            "e_rv":   e_rv,
            "bjd_min": bjd.min(),
            "bjd_max": bjd.max(),
            "n":      len(bjd),
        })
        print(f"  [{owner:6s}]  {short_label(fpath):55s}  "
              f"N={len(bjd):4d}  BJD [{bjd.min():.2f} – {bjd.max():.2f}]")
    return catalog

ashley_cat = build_catalog(ashley_files, "Ashley")
chris_cat  = build_catalog(chris_files,  "Chris")

# ── Overlap detection ─────────────────────────────────────────────────────────

def ranges_overlap(a_min, a_max, c_min, c_max):
    """True if two BJD intervals intersect."""
    return a_min <= c_max and c_min <= a_max


def count_shared_epochs(bjd_a, bjd_c, tol=EPOCH_MATCH_TOL):
    """Count BJD points that are within `tol` days of each other."""
    shared = 0
    for t in bjd_a:
        if np.any(np.abs(bjd_c - t) <= tol):
            shared += 1
    return shared


print("\n" + "─" * 80)
print("OVERLAP ANALYSIS")
print("─" * 80)

overlaps = []   # list of dicts describing each overlapping pair

for a in ashley_cat:
    for c in chris_cat:
        if not ranges_overlap(a["bjd_min"], a["bjd_max"],
                              c["bjd_min"], c["bjd_max"]):
            continue

        # Overlap interval
        ov_start = max(a["bjd_min"], c["bjd_min"])
        ov_end   = min(a["bjd_max"], c["bjd_max"])
        ov_span  = ov_end - ov_start

        # Points each dataset has inside the overlap window
        a_in_ov = np.sum((a["bjd"] >= ov_start) & (a["bjd"] <= ov_end))
        c_in_ov = np.sum((c["bjd"] >= ov_start) & (c["bjd"] <= ov_end))

        # Shared epochs (same night / same observation)
        shared = count_shared_epochs(a["bjd"], c["bjd"])

        overlaps.append({
            "ashley":    a,
            "chris":     c,
            "ov_start":  ov_start,
            "ov_end":    ov_end,
            "ov_span":   ov_span,
            "a_pts_in":  a_in_ov,
            "c_pts_in":  c_in_ov,
            "shared":    shared,
        })

if not overlaps:
    print("No BJD range overlaps found between Ashley and Chris datasets.")
else:
    print(f"Found {len(overlaps)} overlapping file pair(s):\n")
    header = (f"{'Ashley file':45s}  {'Chris file':45s}  "
              f"{'Overlap start':>14}  {'Overlap end':>12}  "
              f"{'Span(d)':>8}  {'A pts':>6}  {'C pts':>6}  {'Shared':>6}")
    print(header)
    print("─" * len(header))
    for ov in overlaps:
        print(f"{ov['ashley']['label']:45s}  {ov['chris']['label']:45s}  "
              f"{ov['ov_start']:14.4f}  {ov['ov_end']:12.4f}  "
              f"{ov['ov_span']:8.1f}  {ov['a_pts_in']:6d}  "
              f"{ov['c_pts_in']:6d}  {ov['shared']:6d}")

# ── Plot 1 : Timeline / Gantt ─────────────────────────────────────────────────

all_entries = ashley_cat + chris_cat
all_entries_sorted = sorted(all_entries, key=lambda x: x["bjd_min"])

fig1, ax1 = plt.subplots(figsize=(16, max(6, len(all_entries_sorted) * 0.35)))

COLORS = {"Ashley": "#E07B54", "Chris": "#4F8BC9"}
y_ticks, y_labels = [], []

for i, entry in enumerate(all_entries_sorted):
    color  = COLORS[entry["owner"]]
    y      = i
    ax1.barh(y,
             entry["bjd_max"] - entry["bjd_min"],
             left=entry["bjd_min"],
             height=0.6, color=color, alpha=0.75, edgecolor="k", linewidth=0.4)

    # Scatter the actual BJD points as ticks along the bar
    ax1.scatter(entry["bjd"], np.full_like(entry["bjd"], y),
                color="k", s=3, zorder=3, alpha=0.4)

    y_ticks.append(y)
    y_labels.append(f"[{entry['owner'][0]}] {entry['label']}")

# Shade overlapping regions
for ov in overlaps:
    ax1.axvspan(ov["ov_start"], ov["ov_end"],
                alpha=0.12, color="green", zorder=0)

ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_labels, fontsize=7)
ax1.set_xlabel("BJD", fontsize=12)
ax1.set_title("Dataset Timeline — Ashley [orange] vs Chris [blue]\n"
              "Green shading = overlapping BJD windows", fontsize=12)
ax1.grid(True, axis="x", linestyle="--", alpha=0.4)

legend_patches = [mpatches.Patch(color=COLORS["Ashley"], label="Ashley"),
                  mpatches.Patch(color=COLORS["Chris"],  label="Chris"),
                  mpatches.Patch(color="green", alpha=0.3, label="Overlap window")]
ax1.legend(handles=legend_patches, loc="lower right", fontsize=9)
fig1.tight_layout()

# ── Plot 2 : e_RV vs BJD — overlapping pairs only ────────────────────────────

if overlaps:
    fig2, ax2 = plt.subplots(figsize=(14, 6))

    plotted_ashley = set()
    plotted_chris  = set()

    for ov in overlaps:
        a, c = ov["ashley"], ov["chris"]

        if a["label"] not in plotted_ashley:
            ax2.scatter(a["bjd"], a["e_rv"],
                        color=COLORS["Ashley"], s=15, alpha=0.8,
                        edgecolors="k", linewidths=0.3, zorder=3,
                        label=f"[A] {a['label']}")
            plotted_ashley.add(a["label"])

        if c["label"] not in plotted_chris:
            ax2.scatter(c["bjd"], c["e_rv"],
                        color=COLORS["Chris"], marker="^", s=35, alpha=0.8,
                        edgecolors="k", linewidths=0.3, zorder=3,
                        label=f"[C] {c['label']}")
            plotted_chris.add(c["label"])

        ax2.axvspan(ov["ov_start"], ov["ov_end"],
                    alpha=0.08, color="green", zorder=0)

    ax2.set_xlabel("BJD", fontsize=12)
    ax2.set_ylabel("e_RV  [m/s]", fontsize=12)
    ax2.set_xlim(2456000, 2458000)
    ax2.set_title("e_RV vs Time",
                  fontsize=12)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
               fontsize=8, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.4)
    fig2.tight_layout()

    # ── Plot 3 : RV vs BJD — overlapping pairs only ──────────────────────────

    fig3, ax3 = plt.subplots(figsize=(14, 6))
    plotted_ashley, plotted_chris = set(), set()

    for ov in overlaps:
        a, c = ov["ashley"], ov["chris"]

        if a["label"] not in plotted_ashley:
            ax3.errorbar(a["bjd"], a["rv"], yerr=a["e_rv"],
                         fmt="o", ms=1, alpha=1,
                         color=COLORS["Ashley"], elinewidth=0.8, capsize=2,
                         label=f"[A] {a['label']}")
            plotted_ashley.add(a["label"])

        if c["label"] not in plotted_chris:
            ax3.errorbar(c["bjd"], c["rv"], yerr=c["e_rv"],
                         fmt="^", ms=4, alpha=1,
                         color=COLORS["Chris"], elinewidth=0.8, capsize=2,
                         label=f"[C] {c['label']}")
            plotted_chris.add(c["label"])

        ax3.axvspan(ov["ov_start"], ov["ov_end"],
                    alpha=0.08, color="green", zorder=0)

    ax3.set_xlabel("BJD", fontsize=12)
    ax3.set_ylabel("RV  [m/s]", fontsize=12)
    ax3.set_ylim(-60, 50)
    ax3.set_title("RV vs Time",
                  fontsize=12)
    ax3.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
               fontsize=8, framealpha=0.9)
    ax3.grid(True, linestyle="--", alpha=0.4)
    fig3.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────

out_dir = os.path.dirname(os.path.abspath(__file__))

saves = [(fig1, "overlap_timeline.png")]
if overlaps:
    saves += [(fig2, "overlap_eRV_vs_time.png"),
              (fig3, "overlap_RV_vs_time.png")]

for fig, name in saves:
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {path}")

plt.show()

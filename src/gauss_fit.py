"""
Baseline (prior art): fit 5 Gaussians (P, Q, R, S, T) to each beat by least squares.
No neural network, no differential equation. Used to test whether the PINN adds value.

Uses exactly the same R-peaks, beat boundaries, band-pass and amplitude scale as the
PINN run, by reading its beat_params CSV.

Usage:
    python src/gauss_fit.py --labels data/cls/labels.csv --pinn outputs/cls --out outputs/cls_gauss
Output: outputs/cls_gauss/beat_params_gauss.csv  (same column names as the PINN CSV:
        P_a = height, P_b = width, P_theta = center, beat_mse = fit error)
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import wfdb
from scipy.optimize import least_squares

from physics import DEFAULT_B, DEFAULT_THETA, WAVES, simulate
from train import bandpass

# Allowed center range (rad) for each wave, R-peak at 0
CENTER_BOUNDS = [(-np.pi, -0.4), (-0.6, 0.0), (-0.2, 0.2), (0.0, 0.6), (0.6, np.pi)]


def model(p, th):
    z = np.full_like(th, p[-1])
    for i in range(5):
        h, mu, w = p[3 * i: 3 * i + 3]
        z += h * np.exp(-(th - mu) ** 2 / (2 * w ** 2))
    return z


def fit_beat(th, z):
    p0, lo, hi = [], [], []
    for i in range(5):
        mu0 = DEFAULT_THETA[i]
        h0 = float(np.interp(mu0, th, z)) if th.min() <= mu0 <= th.max() else 0.0
        p0 += [h0, mu0, DEFAULT_B[i]]
        lo += [-np.inf, CENTER_BOUNDS[i][0], 0.02]
        hi += [np.inf, CENTER_BOUNDS[i][1], 1.0]
    p0 += [0.0]; lo += [-np.inf]; hi += [np.inf]
    p0 = np.clip(p0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
    res = least_squares(lambda p: model(p, th) - z, p0, bounds=(lo, hi), method="trf", max_nfev=2000)
    return res.x, float(np.mean(res.fun ** 2))


def process(path, beats):
    rec = wfdb.rdrecord(path)
    fs = rec.fs
    lead = beats.lead.iloc[0]
    x = bandpass(np.nan_to_num(rec.p_signal[:, rec.sig_name.index(lead)]), fs)
    t_all = np.arange(len(x)) / fs
    r = beats.sort_values("beat").r_time_s.values
    rr = np.diff(r)
    # Same scale as extract_beats.py
    ref = np.ptp(simulate(np.arange(0, 3, 1 / fs), 2 * np.pi / rr.mean(), 0.5))
    beat_ptp = np.median([np.ptp(x[(t_all > a) & (t_all < b)]) for a, b in zip(r[:-1], r[1:])])
    z_all = x * ref / beat_ptp

    rows = []
    for k in range(len(r)):
        rr_prev = rr[k - 1] if k > 0 else rr[0]
        rr_next = rr[k] if k < len(rr) else rr[-1]
        sel = (t_all >= r[k] - rr_prev / 2) & (t_all < r[k] + rr_next / 2)
        t = t_all[sel]
        # Same phase convention as BeatPINN: before the R-peak use the previous RR, after it the next RR
        th = np.where(t < r[k], 2 * np.pi * (t - r[k]) / rr_prev, 2 * np.pi * (t - r[k]) / rr_next)
        p, mse = fit_beat(th, z_all[sel])
        row = {"record": beats.record.iloc[0], "lead": lead, "beat": k, "r_time_s": r[k],
               "beat_mse": mse, "z0": p[-1]}
        for i, w in enumerate(WAVES):
            row[f"{w}_a"], row[f"{w}_theta"], row[f"{w}_b"] = p[3 * i: 3 * i + 3]
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/cls/labels.csv")
    ap.add_argument("--data", default="data/cls")
    ap.add_argument("--pinn", default="outputs/cls")
    ap.add_argument("--out", default="outputs/cls_gauss")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lab = pd.read_csv(args.labels, dtype=str)
    paths = {s: os.path.join(args.data, p.replace("files/", "")) for s, p in zip(lab.study_id, lab.path)}
    pinn = pd.concat([pd.read_csv(f, dtype={"record": str})
                      for f in glob.glob(os.path.join(args.pinn, "beat_params*.csv"))])
    rows, t0 = [], time.time()
    for n, (rec, b) in enumerate(pinn.groupby("record"), 1):
        rows += process(paths[rec], b)
        if n % 50 == 0:
            print(f"{n} records, {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "beat_params_gauss.csv"), index=False)
    print(f"done: {len(rows)} beats, {pinn.record.nunique()} records, {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()

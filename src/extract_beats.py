"""
Beat-by-beat physical parameter extraction for MIMIC-IV-ECG records.

For each record: one PINN is trained on the full 10-second lead, while the McSharry
parameters (a, b, theta for P, Q, R, S, T) are estimated separately for every beat.

Usage:
    python src/download_sample.py --n 20
    python src/extract_beats.py --data data/mimic-iv-ecg --lead II

Input  : WFDB records found under --data (one lead each)
Output : outputs/beat_params.csv      (one row per beat)
         outputs/skipped_records.csv  (records that could not be processed, with reason)
         outputs/beat_figs/<record>.png
"""
import argparse
import glob
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import find_peaks

from physics import WAVES, simulate
from pinn import BeatPINN, train
from train import bandpass


def detect_r_peaks(x, fs):
    """R-peaks on a band-passed lead. Returns peak times (s)."""
    peaks, _ = find_peaks(x, distance=int(0.3 * fs), prominence=0.5 * np.percentile(np.abs(x), 99))
    return peaks / fs


def subject_id(rec):
    for c in rec.comments:
        if "subject_id" in c:
            return c.split(":")[-1].strip()
    return ""


def process_record(path, lead, epochs, lam, fig_dir):
    rec = wfdb.rdrecord(path)
    if lead not in rec.sig_name:
        raise ValueError(f"lead {lead} not found")
    fs = rec.fs
    x = bandpass(np.nan_to_num(rec.p_signal[:, rec.sig_name.index(lead)]), fs)
    t_all = np.arange(len(x)) / fs

    r = detect_r_peaks(x, fs)
    if len(r) < 4:
        raise ValueError(f"only {len(r)} R-peaks detected")
    rr = np.diff(r)
    hr = 60 / rr
    if hr.min() < 30 or hr.max() > 200:
        raise ValueError(f"implausible heart rate {hr.min():.0f}-{hr.max():.0f} bpm")

    # Window: half a beat before the first R-peak to half a beat after the last one
    lo, hi = r[0] - rr[0] / 2, r[-1] + rr[-1] / 2
    m = (t_all >= max(lo, 0)) & (t_all <= min(hi, t_all[-1]))
    t, z = t_all[m], x[m]

    # Scale to the default amplitude of the physics model (same idea as train.py)
    omega_mean = 2 * np.pi / rr.mean()
    tt = np.arange(0, 3, 1 / fs)
    ref = np.ptp(simulate(tt, omega_mean, 0.5))
    beat_ptp = np.median([np.ptp(x[(t_all > a_) & (t_all < b_)]) for a_, b_ in zip(r[:-1], r[1:])])
    scale = ref / beat_ptp
    z = z * scale

    model = BeatPINN(r, (t[0], t[-1]))
    t_col = np.random.default_rng(1).uniform(t[0], t[-1], 4000)
    hist = train(model, t, z, t_col, epochs=epochs, lam=lam, log_every=epochs)

    with torch.no_grad():
        tt_ = torch.tensor(t, dtype=torch.float32).view(-1, 1)
        z_hat = model(tt_).numpy().ravel()
        _, _, j = model.phase(tt_)
        j = j.numpy()

    a, b, th = model.beat_params()
    rows = []
    name = os.path.basename(path)
    sid = subject_id(rec)
    for k in range(model.n_beats):
        sel = j == k
        row = {
            "record": name, "subject_id": sid, "lead": lead, "beat": k,
            "r_time_s": r[k],
            "rr_s": rr[k] if k < len(rr) else rr[-1],
            "hr_bpm": 60 / (rr[k] if k < len(rr) else rr[-1]),
            "beat_mse": float(np.mean((z_hat[sel] - z[sel]) ** 2)) if sel.any() else np.nan,
        }
        for i, w in enumerate(WAVES):
            row[f"{w}_a"], row[f"{w}_b"], row[f"{w}_theta"] = a[k, i], b[k, i], th[k, i]
        row |= {"z0": float(model.z0.detach()), "scale": scale,
                "record_data_loss": hist[-1][0], "record_physics_loss": hist[-1][1]}
        rows.append(row)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, z, "k", lw=1, alpha=.6, label="measured (scaled)")
    ax.plot(t, z_hat, "C3", lw=1.2, label="PINN")
    for rk in r:
        ax.axvline(rk, color="C0", lw=.5, alpha=.4)
    ax.set_title(f"{name} {lead} | {model.n_beats} beats | HR {np.mean(hr):.0f} bpm")
    ax.set_xlabel("time (s)"); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f"{name}.png"), dpi=100); plt.close(fig)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mimic-iv-ecg", help="folder searched for .hea files")
    ap.add_argument("--lead", default="II")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lam", type=float, default=1e-3, help="weight of the physics loss")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th record starting at i (parallel runs)")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads per process")
    args = ap.parse_args()
    torch.manual_seed(0)
    if args.threads:
        torch.set_num_threads(args.threads)
    shard_i, shard_n = map(int, args.shard.split("/"))
    tag = "" if shard_n == 1 else f"_shard{shard_i}"

    fig_dir = os.path.join(args.out, "beat_figs")
    os.makedirs(fig_dir, exist_ok=True)
    paths = sorted(p[:-4] for p in glob.glob(os.path.join(args.data, "**", "*.hea"), recursive=True))
    if args.max_records:
        paths = paths[: args.max_records]
    paths = paths[shard_i::shard_n]
    print(f"{len(paths)} records found")

    rows, skipped = [], []
    for n, p in enumerate(paths, 1):
        t0 = time.time()
        try:
            r = process_record(p, args.lead, args.epochs, args.lam, fig_dir)
            rows += r
            print(f"[{n}/{len(paths)}] {os.path.basename(p)}: {len(r)} beats ({time.time() - t0:.0f}s)")
        except Exception as e:
            skipped.append({"record": os.path.basename(p), "reason": str(e)})
            print(f"[{n}/{len(paths)}] {os.path.basename(p)}: SKIPPED ({e})")
        # Save after every record so partial results survive an interruption
        pd.DataFrame(rows).to_csv(os.path.join(args.out, f"beat_params{tag}.csv"), index=False)
        pd.DataFrame(skipped, columns=["record", "reason"]).to_csv(
            os.path.join(args.out, f"skipped_records{tag}.csv"), index=False)

    print(f"done: {len(rows)} beats from {len(paths) - len(skipped)} records, {len(skipped)} skipped")


if __name__ == "__main__":
    main()

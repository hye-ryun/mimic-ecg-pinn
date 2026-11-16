"""
ECG PINN 학습 스크립트

1) 합성 데이터(정답 파라미터를 아는 상태)로 검증:
    python src/train.py --synthetic

2) MIMIC-IV-ECG 실제 레코드로 학습:
    python src/train.py --record data/mimic-iv-ecg/p1000/p10000032/s40689238/40689238 --lead II

입력 : ECG 1개 리드의 3초 구간
출력 : outputs/<이름>_fit.png        (측정 vs PINN 재구성 + 물리 손실 곡선)
       outputs/<이름>_params.json    (추정한 P,Q,R,S,T 파라미터, 심박수)
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, filtfilt, find_peaks

from physics import DEFAULT_A, DEFAULT_B, DEFAULT_THETA, WAVES, simulate
from pinn import ECGPINN, train


def bandpass(x, fs, lo=0.5, hi=40.0):
    b, a = butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def estimate_rhythm(x, fs):
    """R-peak 검출 -> 평균 RR 간격으로 omega, 첫 R-peak 시각 t_r 계산."""
    peaks, _ = find_peaks(x, distance=int(0.3 * fs), height=np.percentile(x, 98) * 0.5)
    rr = np.diff(peaks).mean() / fs
    return 2 * np.pi / rr, peaks[0] / fs, 60.0 / rr


def load_mimic(record, lead, seconds):
    import wfdb
    rec = wfdb.rdrecord(record)
    sig = rec.p_signal[:, rec.sig_name.index(lead)]
    sig = np.nan_to_num(sig)
    fs = rec.fs
    x = bandpass(sig, fs)[: int(seconds * fs)]
    return np.arange(len(x)) / fs, x, fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--record", help="확장자 없는 WFDB 경로")
    ap.add_argument("--lead", default="II")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lam", type=float, default=1e-3, help="물리 손실 가중치")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    truth = None
    if args.synthetic:
        fs, hr = 250, 72.0
        omega = 2 * np.pi * hr / 60
        t = np.arange(0, args.seconds, 1 / fs)
        a_true = DEFAULT_A * np.array([1.5, 1.0, 1.0, 0.8, 1.6])   # 기본값과 다른 '정답'
        z = simulate(t, omega, t_r=0.3, a=a_true)
        z = z + np.random.default_rng(0).normal(0, 0.02 * np.ptp(z), len(z))  # 측정 잡음
        truth = {w: float(a_true[i]) for i, w in enumerate(WAVES)}
        name = "synthetic"
        omega, t_r, hr = estimate_rhythm(z, fs)
        scale = 1.0
    else:
        t, z, fs = load_mimic(args.record, args.lead, args.seconds)
        omega, t_r, hr = estimate_rhythm(z, fs)
        # 실제 ECG(mV)를 물리 모델 기본 진폭에 맞게 스케일
        ref = simulate(t, omega, t_r)
        scale = np.ptp(ref) / np.ptp(z)
        z = z * scale
        name = os.path.basename(args.record) + "_" + args.lead

    print(f"HR ≈ {hr:.1f} bpm, omega = {omega:.3f} rad/s, first R = {t_r:.3f}s")

    model = ECGPINN(omega, t_r)
    t_col = np.random.default_rng(1).uniform(t[0], t[-1], 2000)
    hist = train(model, t, z, t_col, epochs=args.epochs, lam=args.lam)

    with torch.no_grad():
        z_hat = model(torch.tensor(t, dtype=torch.float32).view(-1, 1)).numpy().ravel()
    params = model.params_dict()
    phys_sim = simulate(t, omega, t_r,
                        a=np.array([params[w]["a"] for w in WAVES]),
                        b=np.array([params[w]["b"] for w in WAVES]),
                        theta=np.array([params[w]["theta_rad"] for w in WAVES]),
                        z0=params["z0"])

    result = {"heart_rate_bpm": hr, "amplitude_scale": scale, "waves": params}
    if truth:
        result["true_a"] = truth
    with open(os.path.join(args.out, f"{name}_params.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(2, 1, figsize=(11, 6))
    ax[0].plot(t, z, "k", lw=1, alpha=.6, label="measured")
    ax[0].plot(t, z_hat, "C3", lw=1.5, label="PINN")
    ax[0].plot(t, phys_sim, "C0--", lw=1, label="physics model (learned params)")
    ax[0].set_title(f"{name}  |  HR {hr:.0f} bpm"); ax[0].legend(); ax[0].set_xlabel("time (s)")
    h = np.array(hist)
    ax[1].semilogy(h[:, 0], label="data loss"); ax[1].semilogy(h[:, 1], label="physics loss")
    ax[1].legend(); ax[1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"{name}_fit.png"), dpi=120)
    print("saved to", args.out)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

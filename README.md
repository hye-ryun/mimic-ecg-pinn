# MIMIC-ECG PINN — Physics-Informed Neural Network for ECG

**English** | [한국어](README.ko.md)

This project applies a **Physics-Informed Neural Network (PINN)** to 12-lead ECGs from MIMIC-IV-ECG.
The network reconstructs the ECG while obeying a cardiac dynamical model (the McSharry ECG model).
In doing so, it **recovers interpretable physiological parameters** (amplitude, width and timing) for the
P, Q, R, S and T waves, which makes this an inverse problem.

![synthetic validation](docs/synthetic_fit.png)

## Approach

| Component | Description |
|---|---|
| Input | Time `t` (one ECG lead, 3-second window) |
| Output | ECG voltage `z(t)` |
| Physics | `dz/dt = -Σ a_i·Δθ_i·ω·exp(-Δθ_i² / 2b_i²) − (z − z0)` (McSharry et al., 2003) |
| Loss | Data loss (MSE) + λ × physics loss (ODE residual, derivatives via autograd) |
| Trainable | Network weights + physical parameters `a_i, b_i, θ_i, z0` |
| Results | Reconstruction plot + per-wave parameter JSON (interpretable physiological markers) |

Here `θ_i` is the phase position of each wave (R = 0), `a_i` its amplitude, `b_i` its width,
and `ω = 2π / RR` the angular frequency estimated from detected R-peaks.

## Repository structure

```
mimic-ecg-pinn/
├── src/
│   ├── physics.py          # McSharry cardiac dynamical model (ODE, simulation)
│   ├── pinn.py             # PINN models (single window / beat-by-beat) + training loop
│   ├── download_sample.py  # Download sample records from MIMIC-IV-ECG
│   ├── train.py            # Single-window fit (synthetic / real data)
│   ├── extract_beats.py    # Beat-by-beat parameter extraction over many records
│   ├── select_records.py   # Pick NORMAL / PVC / AF records using MIMIC machine reports
│   ├── classify.py         # NORMAL / PVC / AF classification from beat parameters
│   ├── gauss_fit.py        # Baseline: plain 5-Gaussian curve fit per beat (no NN, no ODE)
│   └── compare_methods.py  # PINN vs Gaussian-fit features, same classifiers and CV
├── docs/                   # Figures for the README
├── data/                   # (git-ignored) downloaded ECGs
└── outputs/                # (git-ignored) training results
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Validate on synthetic data (ground-truth parameters are known)
python src/train.py --synthetic

# 2) Download MIMIC-IV-ECG samples (Open Access, no account required)
python src/download_sample.py --n 3

# 3) Train on a real ECG
python src/train.py --record data/mimic-iv-ecg/p1000/p10000032/s40689238/40689238 --lead II

# 4) Beat-by-beat parameters for every downloaded record
python src/download_sample.py --n 20
python src/extract_beats.py --lead II

# 5) Classification: NORMAL / PVC / AF (needs machine_measurements.csv, 183 MB, from PhysioNet)
python src/select_records.py --per-class 100 --download
python src/extract_beats.py --data data/cls --out outputs/cls --shard 0/3 --threads 3   # also run 1/3 and 2/3 in parallel
python src/classify.py
python src/gauss_fit.py && python src/compare_methods.py
```

3,000 epochs take about 20–30 seconds on CPU (Apple M1/M2).

## Beat-by-beat extraction

`extract_beats.py` trains **one PINN per 10-second record** but gives **each beat its own physical
parameters** (a, b, θ for P, Q, R, S, T). The phase runs linearly from one detected R-peak to the next,
so heart-rate changes between beats are handled.

| Output | Content |
|---|---|
| `outputs/beat_params.csv` | One row per beat: `record, subject_id, lead, beat, r_time_s, rr_s, hr_bpm, beat_mse, P_a, P_b, P_theta, …, T_theta, z0, scale, record_data_loss, record_physics_loss` |
| `outputs/skipped_records.csv` | Records that could not be processed, with the reason (e.g. too few R-peaks) |
| `outputs/beat_figs/<record>.png` | Measured vs PINN reconstruction with R-peaks marked |

**First run (20 records from 7 subjects, lead II, 3,000 epochs):** 271 beats extracted, 0 records skipped,
about 40 s per record on CPU. Heart rate ranged 47–199 bpm (mean 88). The amplitude of the T wave varied
by a median of 16% (coefficient of variation) from beat to beat within a record.

**Observation:** the beats with the largest reconstruction error (`beat_mse`) were **premature (ectopic)
beats**: early, wide QRS complexes followed by a pause. The McSharry model describes a normal P-QRS-T
shape, so it does not fit these beats well. Using model-fit residuals to detect premature beats is an
established idea (e.g. Sayadi et al., 2010, with a Kalman-filter wave model); here it shows up in the PINN too.

## Classification: NORMAL / PVC / AF

**Setup.** Records were labeled from the MIMIC-IV-ECG machine reports (`select_records.py`): NORMAL
(sinus rhythm, "normal ECG", nothing abnormal), PVC (sinus rhythm with ventricular premature complexes)
and AF (atrial fibrillation), excluding flutter, pacing and bundle branch block. 100 records per class,
one study per subject. 12 records were skipped because R-peaks could not be detected
(AF 7, NORMAL 3, PVC 2), leaving **288 records** (NORMAL 97, PVC 98, AF 93, lead II, 4,014 beats).

Each record is summarized by rhythm features (RR mean, SD, CV, RMSSD) and by the median and SD of each
beat parameter plus fit-error summaries. Evaluation: 5x repeated stratified 5-fold CV, macro one-vs-rest AUC.

| Features | Random forest | Logistic regression |
|---|---|---|
| RR only (baseline) | 0.845 | 0.868 |
| Gaussian-fit parameters only | 0.902 | 0.872 |
| PINN parameters only | 0.925 | 0.908 |
| RR + Gaussian fit | 0.918 | 0.893 |
| **RR + PINN** | **0.936** | **0.914** |
| RR + PINN + Gaussian fit | 0.946 | 0.910 |

Per-class AUC (random forest, RR + PINN): NORMAL 0.981, PVC 0.897, AF 0.930.
With RR only, PVC and AF were hard to separate (0.755 and 0.811), since both have irregular RR intervals.

**The parameters behave as physiology predicts.** Median P-wave amplitude `P_a` was 1.26 in NORMAL,
0.76 in PVC and 0.25 in AF (AF has no organized P wave). The ratio of the worst beat's fit error to the
typical beat's (`mse_max_over_med`, log10) was 0.31 in NORMAL, 0.39 in AF and 0.98 in PVC (a few beats
with a different shape).

![classification features](docs/classification_features.png)

**PINN vs plain curve fitting.** The key question is whether the PINN adds anything over the established
approach of fitting 5 Gaussians to each beat (Clifford et al., 2005). Both methods were run on the same
beats with the same features, classifiers and CV. A threshold was set **before** running: the PINN counts
as better only if RR + PINN beats RR + Gaussian by at least +0.02 AUC (random forest).

- Result: 0.936 vs 0.918 = **+0.018, below the threshold.** The PINN was ahead in all 5 CV repeats
  (+0.008 to +0.026) and by +0.023 to +0.036 when shape parameters are used alone, with the largest gain for PVC.
- The Gaussian fit matches each beat more closely (median beat MSE 1.0e-4 vs 4.1e-4 for the PINN), yet its
  parameters classify slightly worse.
- The PINN's physics-loss value itself adds nothing (0.935 without it).
- Compute: Gaussian fit about 7 minutes in total; PINN about 70 minutes on 3 parallel processes (~30x more CPU).

**Conclusion:** PINN beat parameters give a small but consistent advantage over plain curve fitting, not
large enough to meet the pre-set threshold.

**Limitations**
- Labels are machine reports, not cardiologist annotations, and the machine algorithm itself uses rhythm
  and morphology, so some circularity is possible.
- The rhythm baseline is simple (4 RR statistics). Standard interval measurements (PR, QRS, QT) were not
  included as a baseline.
- Single lead (II), 288 records, one dataset. CV repeats are not independent samples, so no significance test
  was done.

## Synthetic validation (wave amplitudes a_i)

| Wave | Ground truth | PINN estimate |
|---|---|---|
| P | 1.80 | 1.85 |
| Q | -5.00 | -4.84 |
| R | 30.00 | 29.70 |
| S | -6.00 | -7.49 |
| T | 1.20 | 1.19 |

The S wave overlaps in time with the Q and R waves. This makes its parameters harder to identify
(low identifiability) and is a target for future work.

## Data

- **MIMIC-IV-ECG v1.0** (PhysioNet, Open Data Commons Open Database License v1.0)
  Gow B, et al. *MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset.* PhysioNet.
  https://physionet.org/content/mimic-iv-ecg/1.0/
- No data files are included in this repository. Download them with `download_sample.py`.

## References

- McSharry PE, Clifford GD, Tarassenko L, Smith LA. A dynamical model for generating synthetic
  electrocardiogram signals. *IEEE Trans Biomed Eng.* 2003;50(3):289–294.
- Raissi M, Perdikaris P, Karniadakis GE. Physics-informed neural networks. *J Comput Phys.* 2019;378:686–707.
- Clifford GD, Shoeb A, McSharry PE, Janz BA. Model-based filtering, compression and classification of the ECG.
  *Int J Bioelectromagnetism.* 2005;7(1):158–161.
- Sayadi O, Shamsollahi MB, Clifford GD. Robust detection of premature ventricular contractions using a
  wave-based Bayesian framework. *IEEE Trans Biomed Eng.* 2010;57(2):353–362.

## Roadmap

- [x] Beat-by-beat parameter estimation, to track how parameters change over time
- [ ] Joint training on all 12 leads (3D cardiac vector model)
- [x] Diagnostic / arrhythmia classification using the estimated parameters as features (linked to MIMIC-IV machine_measurements)
- [ ] Compare against standard interval measurements (PR, QRS, QT) and cardiologist-labeled data
- [ ] Extension to a PINN based on a cardiac electrophysiology PDE (Aliev–Panfilov)

## License

Code: [MIT](LICENSE). The MIMIC-IV-ECG data is not included and is governed by its own license (ODbL v1.0).

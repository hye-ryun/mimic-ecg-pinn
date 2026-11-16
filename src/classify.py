"""
Classify NORMAL / PVC / AF records from beat-by-beat PINN parameters.

Compares three feature sets with repeated stratified 5-fold cross-validation:
  RR      : rhythm only (RR mean, SD, CV, RMSSD)                    <- baseline
  PINN    : per-record summaries of beat parameters + fit error
  RR+PINN : both

Usage:
    python src/classify.py --labels data/cls/labels.csv --beats outputs/cls
Output (in --beats folder): classification_summary.txt, features.csv,
                            feature_boxplots.png, confusion_matrix.png
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WAVES = ["P", "Q", "R", "S", "T"]
CLASSES = ["NORMAL", "PVC", "AF"]


def record_features(b):
    b = b.sort_values("beat")
    rr = np.diff(b.r_time_s.values)
    f = {"rr_mean": rr.mean(), "rr_std": rr.std(), "rr_cv": rr.std() / rr.mean(),
         "rmssd": np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 1 else np.nan,
         "n_beats": len(b)}
    pinn = {}
    for w in WAVES:
        for p in ("a", "b", "theta"):
            v = b[f"{w}_{p}"]
            pinn[f"{w}_{p}_med"] = v.median()
            pinn[f"{w}_{p}_std"] = v.std()
    mse = b.beat_mse
    pinn |= {"mse_mean": np.log10(mse.mean()), "mse_max": np.log10(mse.max()),
             "mse_max_over_med": np.log10(mse.max() / mse.median()),
             "physics_loss": np.log10(b.record_physics_loss.iloc[0])}
    return f, pinn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/cls/labels.csv")
    ap.add_argument("--beats", default="outputs/cls")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    lab = pd.read_csv(args.labels, dtype=str)
    beats = pd.concat([pd.read_csv(f, dtype={"record": str})
                       for f in glob.glob(os.path.join(args.beats, "beat_params*.csv"))])
    skipped = pd.concat([pd.read_csv(f, dtype={"record": str})
                         for f in glob.glob(os.path.join(args.beats, "skipped_records*.csv"))])

    rows, rr_cols, pinn_cols = [], None, None
    for rec, b in beats.groupby("record"):
        f, pinn = record_features(b)
        rr_cols, pinn_cols = list(f), list(pinn)
        rows.append({"record": rec} | f | pinn)
    X = pd.DataFrame(rows).merge(lab.rename(columns={"study_id": "record"})[["record", "label"]], on="record")
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    X.to_csv(os.path.join(args.beats, "features.csv"), index=False)
    y = X.label.map({c: i for i, c in enumerate(CLASSES)}).values

    out = []
    p = out.append
    p(f"Selected per class: {lab.label.value_counts().to_dict()}")
    sk = skipped.merge(lab.rename(columns={"study_id": "record"}), on="record", how="left")
    p(f"Skipped records: {len(sk)} {sk.label.value_counts().to_dict()}")
    p(f"Used per class: {X.label.value_counts().to_dict()}")
    p("")

    models = {
        "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=0),
    }
    sets = {"RR (baseline)": rr_cols, "PINN only": pinn_cols, "RR + PINN": rr_cols + pinn_cols}
    p(f"Macro one-vs-rest AUC, {args.repeats}x repeated stratified 5-fold CV (mean +/- SD):")
    best = (None, None, -1)
    for mname, mk in models.items():
        for sname, cols in sets.items():
            aucs = []
            for r in range(args.repeats):
                cv = StratifiedKFold(5, shuffle=True, random_state=r)
                prob = cross_val_predict(mk(), X[cols].values, y, cv=cv, method="predict_proba")
                aucs.append(roc_auc_score(y, prob, multi_class="ovr", average="macro"))
            p(f"  {mname:12s} | {sname:14s}: {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
            if np.mean(aucs) > best[2]:
                best = (mname, sname, np.mean(aucs))
    p("")

    # Per-class AUC for each feature set with the best model type
    mname = best[0]
    p(f"Per-class AUC ({mname}, one CV run):")
    probs = {}
    for sname, cols in sets.items():
        prob = cross_val_predict(models[mname](), X[cols].values, y,
                                 cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")
        probs[sname] = prob
        per = [roc_auc_score((y == i).astype(int), prob[:, i]) for i in range(3)]
        p(f"  {sname:14s}: " + "  ".join(f"{c} {v:.3f}" for c, v in zip(CLASSES, per)))
    p("")

    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=0)
    rf.fit(X[rr_cols + pinn_cols].values, y)
    imp = pd.Series(rf.feature_importances_, index=rr_cols + pinn_cols).sort_values(ascending=False)
    p("Top 12 features (random forest impurity importance, RR + PINN, indicative only):")
    for k, v in imp.head(12).items():
        p(f"  {k:20s} {v:.3f}")
    p("")
    key = ["rr_cv", "P_a_med", "P_b_med", "mse_max_over_med", "mse_max", "T_a_med"]
    p("Median of key features by class:")
    p(X.groupby("label")[key].median().reindex(CLASSES).round(3).to_string())

    txt = "\n".join(out)
    print(txt)
    open(os.path.join(args.beats, "classification_summary.txt"), "w").write(txt + "\n")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, k in zip(axes.ravel(), key):
        ax.boxplot([X[X.label == c][k] for c in CLASSES], labels=CLASSES, showfliers=False)
        ax.set_title(k)
    fig.suptitle("Key features by class (MIMIC-IV-ECG, lead II)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.beats, "feature_boxplots.png"), dpi=110)

    cm = confusion_matrix(y, probs["RR + PINN"].argmax(1))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xticks(range(3), CLASSES); ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("machine report label")
    ax.set_title(f"RR + PINN ({mname})")
    fig.tight_layout()
    fig.savefig(os.path.join(args.beats, "confusion_matrix.png"), dpi=110)


if __name__ == "__main__":
    main()

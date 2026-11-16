"""
Compare PINN beat parameters with plain 5-Gaussian curve fitting (prior art) for
NORMAL / PVC / AF classification. Same records, beats, features, classifiers and CV.

Usage:
    python src/compare_methods.py
Output: outputs/cls_gauss/comparison_summary.txt
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from classify import CLASSES, record_features


def features(beats, prefix):
    rows = []
    for rec, b in beats.groupby("record"):
        if "record_physics_loss" not in b:
            b = b.assign(record_physics_loss=np.nan)
        f, shape = record_features(b)
        rows.append({"record": rec} | f | {f"{prefix}{k}": v for k, v in shape.items()})
    return pd.DataFrame(rows), list(f), [f"{prefix}{k}" for k in shape]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/cls/labels.csv")
    ap.add_argument("--pinn", default="outputs/cls")
    ap.add_argument("--gauss", default="outputs/cls_gauss")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    lab = pd.read_csv(args.labels, dtype=str).rename(columns={"study_id": "record"})
    pb = pd.concat([pd.read_csv(f, dtype={"record": str}) for f in glob.glob(os.path.join(args.pinn, "beat_params*.csv"))])
    gb = pd.read_csv(os.path.join(args.gauss, "beat_params_gauss.csv"), dtype={"record": str})

    Fp, rr_cols, pinn_cols = features(pb, "pinn_")
    Fg, _, gauss_cols = features(gb, "gauss_")
    gauss_cols = [c for c in gauss_cols if c != "gauss_physics_loss"]       # no physics loss in curve fitting
    pinn_no_phys = [c for c in pinn_cols if c != "pinn_physics_loss"]
    X = Fp.merge(Fg[["record"] + gauss_cols], on="record").merge(lab[["record", "label"]], on="record")
    X = X.replace([np.inf, -np.inf], np.nan)
    used = rr_cols + pinn_cols + gauss_cols
    n0 = len(X)
    X = X.dropna(subset=used)
    y = X.label.map({c: i for i, c in enumerate(CLASSES)}).values

    out = []
    p = out.append
    p(f"Records: {len(X)} (dropped {n0 - len(X)} with missing features)  {X.label.value_counts().to_dict()}")
    p("Median per-beat fit error (beat_mse): PINN %.2e | Gaussian %.2e"
      % (pb.beat_mse.median(), gb.beat_mse.median()))
    p("")

    models = {
        "RandomForest": lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=0),
        "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)),
    }
    sets = {
        "RR (baseline)": rr_cols,
        "Gaussian only": gauss_cols,
        "PINN only": pinn_cols,
        "RR + Gaussian": rr_cols + gauss_cols,
        "RR + PINN": rr_cols + pinn_cols,
        "RR + PINN (no phys loss)": rr_cols + pinn_no_phys,
        "RR + PINN + Gaussian": rr_cols + pinn_cols + gauss_cols,
    }
    res = {}
    p(f"Macro one-vs-rest AUC, {args.repeats}x repeated stratified 5-fold CV (mean +/- SD):")
    for mname, mk in models.items():
        for sname, cols in sets.items():
            aucs, per = [], []
            for r in range(args.repeats):
                prob = cross_val_predict(mk(), X[cols].values, y, method="predict_proba",
                                         cv=StratifiedKFold(5, shuffle=True, random_state=r))
                aucs.append(roc_auc_score(y, prob, multi_class="ovr", average="macro"))
                per.append([roc_auc_score((y == i).astype(int), prob[:, i]) for i in range(3)])
            res[(mname, sname)] = (np.array(aucs), np.mean(per, axis=0))
            p(f"  {mname:12s} | {sname:26s}: {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}   "
              + " ".join(f"{c} {v:.3f}" for c, v in zip(CLASSES, np.mean(per, axis=0))))
        p("")

    a, b = res[("RandomForest", "RR + PINN")][0], res[("RandomForest", "RR + Gaussian")][0]
    diff = a.mean() - b.mean()
    p("PRIMARY (pre-registered): RandomForest, AUC(RR + PINN) - AUC(RR + Gaussian)")
    p(f"  = {a.mean():.3f} - {b.mean():.3f} = {diff:+.3f}  (per-repeat diffs: "
      + ", ".join(f"{v:+.3f}" for v in a - b) + ")")
    p("  criterion >= +0.02: " + ("MET - PINN adds value" if diff >= 0.02 else "NOT MET"))

    txt = "\n".join(out)
    print(txt)
    open(os.path.join(args.gauss, "comparison_summary.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()

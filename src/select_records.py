"""
Select labeled MIMIC-IV-ECG records for classification, using the machine reports
(report_0 ... report_17 in machine_measurements.csv) as labels.

Labels (mutually exclusive, defined from report text):
  NORMAL : "sinus rhythm" and "normal ECG", and nothing abnormal
  PVC    : sinus rhythm with ventricular premature complexes; no AF/flutter/pacing/bundle branch block
  AF     : atrial fibrillation; no flutter, PVC, pacing or bundle branch block
At most one study per subject, so cross-validation never sees the same person twice.

Usage:
    python src/select_records.py --per-class 100 --download
Input  : data/machine_measurements.csv  (download from PhysioNet, 183 MB)
Output : data/cls/labels.csv  + WFDB files under data/cls/
"""
import argparse
import os
import time
import urllib.request

import pandas as pd

BASE = "https://physionet.org/files/mimic-iv-ecg/1.0"

PVC = r"ventricular premature|premature ventricular|\bpvc"
AF = r"atrial fibrillation"
EXCLUDE_COMMON = r"flutter|pac(?:ed|ing|emaker)|bundle branch block"


def label_reports(path):
    cols = ["subject_id", "study_id"] + [f"report_{i}" for i in range(18)]
    d = pd.read_csv(path, usecols=cols, dtype=str)
    txt = d[[f"report_{i}" for i in range(18)]].fillna("").agg(" | ".join, axis=1).str.lower()
    has = lambda p: txt.str.contains(p)   # noqa: E731
    normal = (has(r"sinus rhythm") & has(r"\bnormal ecg\b")
              & ~has(r"premature|fibrillation|flutter|block|pac(?:ed|ing|emaker)|abnormal"))
    pvc = has(PVC) & has(r"sinus") & ~has(AF) & ~has(EXCLUDE_COMMON)
    af = has(AF) & ~has(PVC) & ~has(EXCLUDE_COMMON)
    d["label"] = None
    d.loc[normal, "label"] = "NORMAL"
    d.loc[pvc, "label"] = "PVC"
    d.loc[af, "label"] = "AF"
    return d.dropna(subset=["label"])[["subject_id", "study_id", "label"]]


def record_path(subject_id, study_id):
    return f"files/p{subject_id[:4]}/p{subject_id}/s{study_id}/{study_id}"


def fetch(url, dst, tries=4):
    for k in range(tries):
        try:
            urllib.request.urlretrieve(url, dst)
            return
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurements", default="data/machine_measurements.csv")
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--out", default="data/cls")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()

    lab = label_reports(args.measurements)
    print("candidates:", lab.label.value_counts().to_dict())
    # one study per subject, then sample per class
    lab = lab.sample(frac=1, random_state=args.seed).drop_duplicates("subject_id")
    pick = (lab.groupby("label", group_keys=False)
               .apply(lambda g: g.sample(min(args.per_class, len(g)), random_state=args.seed)))
    pick["path"] = [record_path(s, t) for s, t in zip(pick.subject_id, pick.study_id)]
    os.makedirs(args.out, exist_ok=True)
    pick.to_csv(os.path.join(args.out, "labels.csv"), index=False)
    print("selected:", pick.label.value_counts().to_dict())

    if args.download:
        for n, p in enumerate(pick.path, 1):
            local = os.path.join(args.out, p.replace("files/", ""))
            os.makedirs(os.path.dirname(local), exist_ok=True)
            for ext in (".hea", ".dat"):
                if not os.path.exists(local + ext):
                    fetch(f"{BASE}/{p}{ext}", local + ext)
            if n % 50 == 0:
                print(f"downloaded {n}/{len(pick)}", flush=True)
        print("download done")


if __name__ == "__main__":
    main()

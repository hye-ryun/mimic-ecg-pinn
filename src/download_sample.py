"""
Download a few sample records from MIMIC-IV-ECG (PhysioNet, Open Access, ODbL).

Usage:
    python src/download_sample.py --n 5

Output: data/mimic-iv-ecg/<p-group>/<subject>/<study>/<record>.hea/.dat
Note: the data/ folder is git-ignored and is never uploaded to GitHub.
"""
import argparse
import os
import urllib.request

DB = "mimic-iv-ecg/1.0"
BASE = f"https://physionet.org/files/{DB}"


def list_records(group="p1000", n=5):
    with urllib.request.urlopen(f"{BASE}/files/{group}/RECORDS", timeout=60) as r:
        lines = r.read().decode().split()
    return [f"files/{group}/{x}" for x in lines[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="number of records to download")
    ap.add_argument("--group", default="p1000", help="one of p1000 to p1999")
    ap.add_argument("--out", default="data/mimic-iv-ecg")
    args = ap.parse_args()

    records = list_records(args.group, args.n)
    for rec in records:
        folder, name = os.path.split(rec)
        out_dir = os.path.join(args.out, folder.replace("files/", ""))
        os.makedirs(out_dir, exist_ok=True)
        for ext in (".hea", ".dat"):
            dst = os.path.join(out_dir, name + ext)
            if not os.path.exists(dst):
                urllib.request.urlretrieve(f"{BASE}/{folder}/{name}{ext}", dst)
        print("downloaded:", os.path.join(out_dir, name))


if __name__ == "__main__":
    main()

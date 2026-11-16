"""
MIMIC-IV-ECG (PhysioNet, Open Access, ODbL) 에서 샘플 레코드 몇 개만 내려받기.

사용법:
    python src/download_sample.py --n 5

결과: data/mimic-iv-ecg/<p폴더>/<환자>/<study>/<record>.hea/.dat
※ data/ 폴더는 .gitignore 로 GitHub에 올라가지 않습니다.
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
    ap.add_argument("--n", type=int, default=5, help="받을 레코드 수")
    ap.add_argument("--group", default="p1000", help="p1000 ~ p1999 중 하나")
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

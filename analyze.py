"""Dataset analysis for ExDark Low-Light Challenge"""
from pathlib import Path
from collections import Counter
import yaml

KIT = Path.home() / ".3lc-kaggle-plugin/data/exdark-low-light/v3/starter_kit"

def count_boxes(split):
    cnt = Counter()
    empty = 0
    total = 0
    label_dir = KIT / f"labels/{split}"
    for txt in label_dir.glob("*.txt"):
        s = txt.read_text().strip()
        if not s:
            empty += 1
            continue
        for line in s.splitlines():
            if not line.strip():
                continue
            cls = int(float(line.split()[0]))
            cnt[cls] += 1
            total += 1
    return cnt, empty, total

if __name__ == "__main__":
    cfg = yaml.safe_load(open(KIT / "dataset.yaml"))
    print("dataset.yaml:", cfg)
    names = cfg["names"]
    print("\nClasses:", names)

    for split in ["train", "val"]:
        cnt, empty, total = count_boxes(split)
        print(f"\n{split}: {total} boxes, {empty} empty files")
        for k in sorted(cnt):
            print(f"  {k} {names[k]}: {cnt[k]}")

    # Image sizes sanity
    import cv2
    train_imgs = list((KIT / "images/train").glob("*.jpg"))
    print(f"\nSample train image sizes (first 3):")
    for p in train_imgs[:3]:
        img = cv2.imread(str(p))
        print(f"  {p.name}: {img.shape}")

    # Submission check
    import csv
    sample = KIT / "sample_submission.csv"
    with open(sample) as f:
        rows = list(csv.DictReader(f))
        print(f"\nsample_submission.csv: {len(rows)} rows, columns {rows[0].keys()}")
        print("First 3:", rows[:3])
        assert rows[0].keys() == {"id", "image_id", "prediction_string"}

    print("\nDone. Next: python train.py --epochs 10")

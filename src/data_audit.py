"""Dataset audit: class imbalance, brightness, box-size traps.

Run: python -m src.data_audit
This is what tells you WHERE the mAP is leaking before you burn GPU hours.
"""
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
import yaml

from .config import KIT

def count_boxes(split: str):
    cnt, empty, total = Counter(), 0, 0
    d = KIT / f"labels/{split}"
    for txt in d.glob("*.txt"):
        s = txt.read_text().strip()
        if not s:
            empty += 1
            continue
        for line in s.splitlines():
            if line.strip():
                cnt[int(float(line.split()[0]))] += 1
                total += 1
    return cnt, empty, total

def brightness_stats(n: int = 200):
    """Mean V channel across a sample — proves how dark ExDark is."""
    import random
    imgs = sorted((KIT / "images/train").glob("*.jpg"))
    random.seed(0)
    vals = []
    for p in random.sample(imgs, min(n, len(imgs))):
        img = cv2.imread(str(p))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        vals.append(float(hsv[:, :, 2].mean()))
    return float(np.mean(vals)), float(np.median(vals)), float(np.percentile(vals, 10))

def box_size_stats(split="train", n=2000):
    import random
    random.seed(1)
    txts = sorted((KIT / f"labels/{split}").glob("*.txt"))
    areas = []
    for t in random.sample(txts, min(n, len(txts))):
        for line in t.read_text().strip().splitlines():
            p = line.split()
            if len(p) >= 5:
                _, _, _, w, h = map(float, p[:5])
                areas.append(w * h)
    areas = np.array(areas)
    return {
        "tiny(<0.1%)": float((areas < 0.001).mean()),
        "small(<1%)": float((areas < 0.01).mean()),
        "median_area": float(np.median(areas)),
    }

def main():
    cfg = yaml.safe_load(open(KIT / "dataset.yaml"))
    print("classes:", cfg["names"])
    for split in ("train", "val"):
        cnt, empty, total = count_boxes(split)
        print(f"\n{split}: {total} boxes, {empty} empty files")
        for k in sorted(cnt):
            print(f"  {k:2d} {cfg['names'][k]:8s} {cnt[k]:5d}")
    mean_v, med_v, p10 = brightness_stats()
    print(f"\nbrightness V mean={mean_v:.1f} median={med_v:.1f} p10={p10:.1f} (0-255; <60 = near-dark)")
    print("box sizes:", box_size_stats())
    print("\nTakeaways:")
    print(" 1. People >> Bus/Cat/Dog -> use CLS_GAIN_RARE + per-class conf (see config.py)")
    print(" 2. Tiny Bottle/Cup boxes -> erasing=0.2 not 0.4, TTA multiscale at inference")
    print(" 3. Dark frames -> hsv_v=0.7 + CLAHE review images for label mining")

if __name__ == "__main__":
    main()

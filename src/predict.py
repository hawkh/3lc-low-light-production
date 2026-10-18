"""Inference on test set -> submission.csv (strictly validated).

Format: id,image_id,prediction_string
prediction_string = "cls conf xc yc w h" repeated, or "no box".
Boxes are YOLO-normalized [0,1]. Each class is cut at its PER_CLASS_CONF.

Usage:
  python -m src.predict --weights runs/detect/prod_v2_fixed/weights/best.pt
"""
import argparse, csv, sys
from pathlib import Path
from ultralytics import YOLO
from .config import KIT, IMGSZ, CLASS_NAMES, PER_CLASS_CONF, DEFAULT_IOU

COLUMNS = {"id", "image_id", "prediction_string"}


def latest_best():
    runs = Path("runs/detect")
    found = sorted(runs.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime) if runs.exists() else []
    return str(found[-1]) if found else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="best.pt; default = newest runs/detect/*/weights/best.pt")
    ap.add_argument("--root", type=Path, default=KIT, help="kit with images/test + sample_submission.csv")
    ap.add_argument("--iou", type=float, default=DEFAULT_IOU)
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    # Raw yolo11n.pt has COCO class ids, which do not match the 12 ExDark ids.
    weights = args.weights or latest_best()
    if weights is None:
        sys.exit("no trained weights: pass --weights or train first")
    model = YOLO(weights)
    print(f"weights: {weights}")

    sample = list(csv.DictReader(open(args.root / "sample_submission.csv")))
    min_conf = min(PER_CLASS_CONF.values())
    rows = []
    for idx, image_id in enumerate(r["image_id"] for r in sample):
        ip = args.root / f"images/test/{image_id}.jpg"
        preds = []
        if ip.exists():
            res = model.predict(str(ip), conf=min_conf, iou=args.iou, imgsz=IMGSZ, verbose=False)[0]
            for b in res.boxes if res.boxes is not None else []:
                cls, conf = int(b.cls.item()), float(b.conf.item())
                if cls not in CLASS_NAMES or conf < PER_CLASS_CONF[cls]:
                    continue
                xc, yc, w, h = b.xywhn.tolist()[0]
                preds.append(f"{cls} {conf:.4f} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        else:
            print(f"  missing test image {ip.name}")
        rows.append((idx, image_id, " ".join(preds) if preds else "no box"))
        if idx % 100 == 0:
            print(f"  {idx}/{len(sample)} {image_id}: {len(preds)} boxes")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "image_id", "prediction_string"])
        w.writerows(rows)

    got = list(csv.DictReader(open(args.out)))
    assert len(got) == len(sample), f"expected {len(sample)} rows, got {len(got)}"
    assert set(got[0].keys()) == COLUMNS
    for r in got:
        if r["prediction_string"] != "no box":
            assert len(r["prediction_string"].split()) % 6 == 0, f"bad token count in {r['image_id']}"
    no_box = sum(r["prediction_string"] == "no box" for r in got)
    print(f"OK {args.out}: {len(got)} rows, {len(got) - no_box} with boxes")
    print(f"submit: kaggle competitions submit -c the-3lc-low-light-detection-challenge -f {args.out} -m \"...\"")


if __name__ == "__main__":
    main()

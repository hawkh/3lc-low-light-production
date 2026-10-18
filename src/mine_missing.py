"""Missing-label miner: confident pred + no GT overlap -> review queue.

This is THE leaderboard lever for this competition (3LC 'The Loop'):
in the dark, annotators miss real objects constantly. The model at 0.8+
confidence with max-IoU < 0.1 vs GT is usually right and the label wrong.

Pipeline:
  1. train baseline (prod_v1, >=5 epochs so preds are sane)
  2. python -m src.mine_missing --weights runs/detect/prod_v1/weights/best.pt --conf 0.8 --out review_queue.json
  3. inspect review/ (CLAHE-brightened crops) — human accepts/rejects
  4. set "status" to "accepted"/"rejected" per entry in review_queue.json
  5. python -m src.apply_fixes --queue review_queue.json --apply
  Next round (conf 0.7, 0.6): mine with --root dataset_fixed, apply with
  --src dataset_fixed --out dataset_fixed_r2.

Bulk-accept without review is prohibited by competition Rules; this script
ENFORCES review by writing a queue, never patching labels directly.
"""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import cv2
from ultralytics import YOLO
from .config import KIT

def box_iou(a, b):
    def xyxy(bx):
        xc, yc, w, h = bx
        return xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
    x1, y1, x2, y2 = xyxy(a)
    xa, ya, xb, yb = xyxy(b)
    iw, ih = max(0, min(x2, xb) - max(x1, xa)), max(0, min(y2, yb) - max(y1, ya))
    inter = iw * ih
    u = (x2 - x1) * (y2 - y1) + (xb - xa) * (yb - ya) - inter
    return inter / u if u > 0 else 0.0

def load_gt(p: Path):
    out = []
    if p.exists() and (s := p.read_text().strip()):
        for line in s.splitlines():
            if line.strip():
                t = line.split()
                out.append((int(float(t[0])), tuple(map(float, t[1:5]))))
    return out

def clahe_save(src: Path, dst: Path):
    img = cv2.imread(str(src))
    if img is None:
        return
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    cv2.imwrite(str(dst), cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.8)
    ap.add_argument("--iou", type=float, default=0.1)
    ap.add_argument("--root", type=Path, default=KIT, help="dataset root (kit or a fixed copy)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="review_queue.json")
    ap.add_argument("--max-images", type=int, default=0, help="0 = all")
    ap.add_argument("--viz", default="review", help="CLAHE crops dir for human review")
    args = ap.parse_args()

    model = YOLO(args.weights)
    imgs = sorted((args.root / f"images/{args.split}").glob("*.jpg"))
    if args.max_images:
        imgs = imgs[: args.max_images]
    viz = Path(args.viz); viz.mkdir(exist_ok=True)

    queue, per_img = [], defaultdict(int)
    for i, ip in enumerate(imgs):
        gt = load_gt(args.root / f"labels/{args.split}/{ip.stem}.txt")
        res = model.predict(str(ip), conf=args.conf, iou=0.5, verbose=False)[0]
        if res.boxes is None:
            continue
        for b in res.boxes:
            cls, conf = int(b.cls.item()), float(b.conf.item())
            try:
                xc, yc, w, h = b.xywhn.tolist()[0]
            except Exception:
                x1, y1, x2, y2 = b.xyxyn.tolist()[0]
                xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            miou = max((box_iou((xc, yc, w, h), g) for _, g in gt), default=0.0)
            if miou < args.iou:
                queue.append({"image": ip.name, "cls": cls,
                              "cls_name": model.names.get(cls, str(cls)),
                              "conf": round(conf, 4), "box": [xc, yc, w, h],
                              "max_iou": round(miou, 4), "gt_count": len(gt),
                              "status": "pending"})
                per_img[ip.name] += 1
        if i % 500 == 0:
            print(f"[{i}/{len(imgs)}] queued={len(queue)}")
    queue.sort(key=lambda q: -q["conf"])
    json.dump(queue, open(args.out, "w"), indent=2)
    with open(Path(args.out).with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "cls", "conf", "xc", "yc", "w", "h", "max_iou", "gt_count"])
        for q in queue:
            w.writerow([q["image"], q["cls"], q["conf"], *q["box"], q["max_iou"], q["gt_count"]])
    # CLAHE crops for top-100 so reviewers can SEE in the dark
    for q in queue[:100]:
        clahe_save(args.root / f"images/{args.split}/{q['image']}", viz / f"{Path(q['image']).stem}_enh.jpg")
    print(f"queued {len(queue)} across {len(per_img)} images -> {args.out} + review/ (top-100 enhanced)")
    print(f"NEXT: open {viz}/, set status accepted/rejected in {args.out}, then run "
          f"python -m src.apply_fixes --queue {args.out} --src {args.root} --apply")

if __name__ == "__main__":
    main()

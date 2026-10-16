"""Data-centric loop: find confident predictions with no GT match (missing labels)

Public solution recipe (3LC The Loop):
- Run trained model on TRAIN split
- For each image, keep predictions with conf > threshold
- Compute IoU vs all GT boxes; if max IoU < 0.1 => candidate missing label
- Sort by confidence descending; inspect top candidates visually
- Bulk-accept without review is prohibited; this script writes candidates for review

Usage: python find_missing.py --run runs/detect/train --conf 0.8 --iou 0.1 --top 200
"""
import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

KIT = Path.home() / ".3lc-kaggle-plugin/data/exdark-low-light/v3/starter_kit"

def iou(box1, box2):
    # boxes in xc,yc,w,h normalized -> xyxy
    def to_xyxy(b):
        xc,yc,w,h = b
        return [xc-w/2, yc-h/2, xc+w/2, yc+h/2]
    x1, y1, x2, y2 = to_xyxy(box1)
    xa1, ya1, xa2, ya2 = to_xyxy(box2)
    xi1, yi1 = max(x1, xa1), max(y1, ya1)
    xi2, yi2 = min(x2, xa2), min(y2, ya2)
    iw, ih = max(0, xi2-xi1), max(0, yi2-yi1)
    inter = iw*ih
    union = (x2-x1)*(y2-y1) + (xa2-xa1)*(ya2-ya1) - inter
    return inter/union if union>0 else 0

def load_gt(label_path):
    boxes=[]
    if not label_path.exists():
        return boxes
    txt=label_path.read_text().strip()
    if not txt:
        return boxes
    for line in txt.splitlines():
        if not line.strip(): continue
        parts=line.split()
        cls=int(float(parts[0])); xc,yc,w,h = map(float, parts[1:5])
        boxes.append((cls, (xc,yc,w,h)))
    return boxes

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="path to train run dir containing weights/best.pt")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--conf", type=float, default=0.8)
    ap.add_argument("--iou", type=float, default=0.1, help="max IoU to consider missing")
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--split", default="train", choices=["train","val"])
    ap.add_argument("--out", default="candidates.json")
    args=ap.parse_args()

    weights = args.weights or str(Path(args.run)/"weights/best.pt")
    if not Path(weights).exists():
        weights = str(Path(args.run)/"weights/best.pt")
        # fallback to last.pt
        if not Path(weights).exists():
            weights = "yolo11n.pt"
    print(f"Using weights: {weights}")
    model = YOLO(weights)

    img_dir = KIT / f"images/{args.split}"
    label_dir = KIT / f"labels/{args.split}"
    images = sorted(img_dir.glob("*.jpg"))[:2000]  # limit for speed; full loop does 5910
    # For full analysis, remove slice above. We default subset for demo, user can run full via --full flag
    # Let's do all if requested
    if len(images)< 5910:
        images = sorted(img_dir.glob("*.jpg"))

    candidates=[]
    for img_path in images:
        label_path = label_dir / f"{img_path.stem}.txt"
        gt_boxes = load_gt(label_path)
        gt_by_cls = {}
        # inference
        results = model.predict(source=str(img_path), conf=args.conf, iou=0.5, verbose=False, save=False)
        for r in results:
            if r.boxes is None or len(r.boxes)==0:
                continue
            for box in r.boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                xc,yc,w,h = map(float, box.xyxyn.tolist()[0] if hasattr(box,'xyxyn') else box.xywhn.tolist()[0])
                # Actually xywhn is xc,yc,w,h normalized
                # Use xywhn
                # Fix extraction: ultralytics box.xywhn is normalized xc yc w h
                try:
                    xc,yc,w,h = box.xywhn.tolist()[0]
                except:
                    # fallback
                    x1,y1,x2,y2 = box.xyxyn.tolist()[0]
                    xc=(x1+x2)/2; yc=(y1+y2)/2; w=x2-x1; h=y2-y1
                pred_box = (xc,yc,w,h)
                # max IoU with any GT
                max_iou = 0
                if gt_boxes:
                    for gcls, gbox in gt_boxes:
                        # class-agnostic IoU check first: if high IoU but wrong class, also candidate? we flag class mismatch separately
                        v = iou(pred_box, gbox)
                        if v>max_iou:
                            max_iou=v
                if max_iou < args.iou:
                    candidates.append({
                        "image": img_path.name,
                        "image_path": str(img_path),
                        "pred_cls": cls,
                        "pred_cls_name": model.names[cls] if hasattr(model,'names') else str(cls),
                        "conf": conf,
                        "box": [xc,yc,w,h],
                        "max_iou": max_iou,
                        "gt_count": len(gt_boxes)
                    })
        if len(candidates) % 20 == 0 and len(images)>100:
            print(f"Scanned {len(candidates)} candidates so far... {img_path.name}")

    candidates.sort(key=lambda x: x["conf"], reverse=True)
    top = candidates[:args.top]
    print(f"\nFound {len(candidates)} candidates with conf>{args.conf} and max IoU<{args.iou}")
    print(f"Top {len(top)} by confidence:")
    for c in top[:10]:
        print(f"  {c['image']} conf {c['conf']:.3f} cls {c['pred_cls_name']} box {c['box']} max_iou {c['max_iou']:.3f} gt {c['gt_count']}")

    # Save JSON + CSV for review
    with open(args.out, "w") as f:
        json.dump(top, f, indent=2)
    csv_out = Path(args.out).with_suffix(".csv")
    with open(csv_out, "w", newline="") as f:
        w=csv.writer(f)
        w.writerow(["image","pred_cls","conf","xc","yc","w","h","max_iou","gt_count"])
        for c in top:
            w.writerow([c["image"], c["pred_cls"], f"{c['conf']:.4f}", *c["box"], f"{c['max_iou']:.4f}", c["gt_count"]])
    print(f"\nSaved {args.out} and {csv_out}")
    print("Next: visually inspect candidates, then run fix_labels.py --apply to add them (per rules, inspect before bulk accept)")

if __name__=="__main__":
    main()

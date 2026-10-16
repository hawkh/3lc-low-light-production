"""Inference on test set -> submission.csv (strictly validated)

Format: id,image_id,prediction_string
prediction_string = "cls conf xc yc w h" repeated or "no box"
Boxes are YOLO normalized [0,1] per competition.
"""
import argparse, csv
from pathlib import Path
from ultralytics import YOLO

KIT = Path.home() / ".3lc-kaggle-plugin/data/exdark-low-light/v3/starter_kit"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="best.pt path; defaults to runs/detect/train*/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--out", default="submission.csv")
    args=ap.parse_args()

    weights = args.weights
    if weights is None:
        # find latest best.pt
        candidates = sorted(Path("runs/detect").glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True) if Path("runs/detect").exists() else []
        if candidates:
            weights = str(candidates[0])
        else:
            weights = "yolo11n.pt"
    print(f"Weights: {weights}")

    model = YOLO(weights)
    print(f"Model names: {model.names}")

    test_dir = KIT / "images/test"
    sample = KIT / "sample_submission.csv"

    # Load sample to get order
    with open(sample) as f:
        rows = list(csv.DictReader(f))
    order = [r["image_id"] for r in rows]
    print(f"Test order: {len(order)} images")

    # Predict
    out_rows=[]
    for idx, image_id in enumerate(order):
        img_path = test_dir / f"{image_id}.jpg"
        if not img_path.exists():
            out_rows.append((idx, image_id, "no box"))
            continue
        results = model.predict(source=str(img_path), conf=args.conf, iou=args.iou, verbose=False)
        preds=[]
        for r in results:
            if r.boxes is None or len(r.boxes)==0:
                continue
            for box in r.boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                # filtered to 0-11 only (plugin does this)
                if cls<0 or cls>11:
                    continue
                try:
                    xc,yc,w,h = box.xywhn.tolist()[0]
                except:
                    x1,y1,x2,y2 = box.xyxyn.tolist()[0]
                    xc=(x1+x2)/2; yc=(y1+y2)/2; w=x2-x1; h=y2-y1
                preds.append(f"{cls} {conf:.4f} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        if preds:
            pred_str = " ".join(preds)
            # Competition expects space-separated groups? Each box is 6 tokens, so joining is fine.
            # But sample shows "class conf xc yc w h" per box concatenated with space
            # We'll produce as space-joined
        else:
            pred_str = "no box"
        out_rows.append((idx, image_id, pred_str))
        if idx % 100 ==0:
            print(f"  {idx}/{len(order)} {image_id}: {len(preds)} boxes")

    with open(args.out, "w", newline="") as f:
        w=csv.writer(f)
        w.writerow(["id","image_id","prediction_string"])
        for r in out_rows:
            w.writerow(r)
    print(f"Wrote {args.out} with {len(out_rows)} rows")
    # Validate
    with open(args.out) as f:
        r=list(csv.DictReader(f))
        assert len(r)==715, f"expected 715 got {len(r)}"
        assert set(r[0].keys())=={"id","image_id","prediction_string"}
        no_box = sum(1 for x in r if x["prediction_string"].strip()=="no box")
        print(f"Validation: {len(r)} rows, {no_box} no-box, {len(r)-no_box} with predictions")
        print("First 3:", r[:3])
        # Check that non-no-box rows have 6-token groups
        for row in r[:5]:
            if row["prediction_string"]!="no box":
                toks=row["prediction_string"].split()
                assert len(toks)%6==0, f"bad token count {len(toks)} in {row}"
    print("Ready to submit: kaggle competitions submit -c the-3lc-low-light-detection-challenge -f submission.csv -m \"message\"")

if __name__=="__main__":
    main()

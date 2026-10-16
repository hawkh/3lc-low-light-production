"""Production training: locked YOLOv11n @640 + low-light recipe + rare-class gains.

Beats vanilla `yolo train` because:
 1. LOWLIGHT_AUG (hsv_v 0.7, erasing 0.2, mixup/copy_paste small) keeps dark
    gradients and tiny Cups alive — vanilla erasing/mosaic deletes them.
 2. cls=0.7 upweights rare Bus/Cat/Dog vs People-dominated CE.
 3. cos_lr + patience + save_period for 4GB-safe long runs.

Usage:
  python -m src.train_production --epochs 20 --batch 4 --name prod_v1
  python -m src.train_production --data dataset_fixed/dataset.yaml --epochs 20 --name prod_v2_fixed
"""
import argparse
from pathlib import Path
from ultralytics import YOLO
from .config import MODEL_ID, IMGSZ, LOWLIGHT_AUG, CLS_GAIN_RARE

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset.yaml")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--name", default="prod_v1")
    ap.add_argument("--device", default=0)
    args = ap.parse_args()

    model = YOLO(MODEL_ID)
    print(f"[train] init={MODEL_ID} imgsz={args.imgsz} epochs={args.epochs} batch={args.batch}")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=2,
        device=args.device,
        amp=True,
        patience=25,
        save_period=5,
        exist_ok=True,
        name=args.name,
        plots=True,
        val=True,
        cls=CLS_GAIN_RARE,
        cos_lr=True,
        seed=0,
        deterministic=True,
        **LOWLIGHT_AUG,
    )
    print("[train] done. best =", f"runs/detect/{args.name}/weights/best.pt")

if __name__ == "__main__":
    main()

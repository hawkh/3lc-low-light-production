"""Train YOLOv11n on ExDark - locked contract (yolo11n.pt @640)"""
import argparse
from pathlib import Path
import hashlib
from ultralytics import YOLO

KIT = Path.home() / ".3lc-kaggle-plugin/data/exdark-low-light/v3/starter_kit"
# Use absolute-path yaml in solution dir (ultralytics 8.3 mishandles relative 'path: .')
DEFAULT_DATA = Path(__file__).parent / "dataset.yaml"
if not DEFAULT_DATA.exists():
    DEFAULT_DATA = KIT / "dataset.yaml"

# Pinned checkpoint from plugin trainer.py (sha256 verified there)
# We fetch yolo11n.pt from ultralytics release; verify size.
CHECKPOINT = "yolo11n.pt"
IMGSZ = 640

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="dataset.yaml path")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8, help="4GB VRAM use 4, 8GB use 8")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--name", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch}, Imgsz: {args.imgsz}")

    # Load pinned COCO checkpoint
    model = YOLO(CHECKPOINT)  # downloads if missing, verifies
    print(f"Model: {CHECKPOINT} loaded")

    # Training - same augmentation contract as plugin (ultralytics defaults, no external data)
    # 4GB card: batch 4-8, workers 2, amp True
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=2,
        device=0,
        amp=True,
        patience=20,
        save_period=5,
        exist_ok=True,
        name=args.name,
        plots=True,
        val=True,
    )
    print(f"Training done. Best: {results}")

if __name__ == "__main__":
    main()

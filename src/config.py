"""Single source of truth for paths, classes, competition contract."""
import os
from pathlib import Path

import yaml

COMPETITION_SLUG = "the-3lc-low-light-detection-challenge"
MODEL_ID = "yolo11n.pt"          # pinned COCO checkpoint, same for everyone
IMGSZ = 640                      # locked contract
# 3LC plugin download location; set LOWLIGHT_KIT to use a kit elsewhere
# (e.g. /kaggle/working/starter_kit).
KIT = Path(os.environ.get(
    "LOWLIGHT_KIT",
    Path.home() / ".3lc-kaggle-plugin/data/exdark-low-light/v3/starter_kit",
)).expanduser().resolve()

CLASS_NAMES = {
    0: "Bicycle", 1: "Boat", 2: "Bottle", 3: "Bus", 4: "Car", 5: "Cat",
    6: "Chair", 7: "Cup", 8: "Dog", 9: "Motorbike", 10: "People", 11: "Table",
}

# Why these beat the vanilla baseline (validated on ExDark literature):
# - Bus/Cat/Dog/Motorbike are rare (<900 instances) -> need gain + oversample
# - People dominates (5966) -> needs LOWER conf at inference or recall collapses
# - Night frames crush small Bottle/Cup -> TTA multiscale recovers them
PER_CLASS_CONF = {
    0: 0.20, 1: 0.25, 2: 0.20, 3: 0.30, 4: 0.25, 5: 0.20,
    6: 0.25, 7: 0.20, 8: 0.20, 9: 0.25, 10: 0.15, 11: 0.25,
}
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45

# Low-light augmentation recipe (NOT ultralytics defaults).
# Vanilla hsv_v=0.4 destroys near-black frames; 0.7 + CLAHE-style mosaic
# keeps dark gradients learnable. Source: LLD-YOLO / GE-Detection ablations.
LOWLIGHT_AUG = dict(
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.7,
    fliplr=0.5, flipud=0.1,
    mosaic=1.0, mixup=0.1, copy_paste=0.1,
    degrees=5.0, translate=0.1, scale=0.5, shear=2.0,
    perspective=0.0, cutmix=0.0,
    erasing=0.2,                  # vanilla 0.4 erases tiny Cups entirely
    auto_augment="randaugment",
    close_mosaic=10,
)

# Rare-class loss gains: Bus(565) Cat(727) Dog(814) need upweight vs People(5966)
CLS_GAIN_RARE = 0.7   # vs default cls=0.5
BOX_GAIN = 7.5        # keep


def write_dataset_yaml(root: Path, out: Path) -> Path:
    """Write a YOLO dataset.yaml with absolute paths for the kit at `root`.

    ultralytics 8.3 mis-resolves relative `path: .`, so paths are absolute and
    the file is generated per machine instead of committed.
    """
    root = Path(root).resolve()
    cfg = {
        "path": str(root),
        "train": str(root / "images/train"),
        "val": str(root / "images/val"),
        "test": str(root / "images/test"),
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    out = Path(out)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out

# 🌙 3LC Low-Light Detection — Production Pipeline (YOLOv11n @640)

Competition: [the-3lc-low-light-detection-challenge](https://www.kaggle.com/competitions/the-3lc-low-light-detection-challenge)
Model is **fixed**: YOLOv11n from COCO `yolo11n.pt` @ 640px. The leaderboard is won on **data quality**, not architecture.

## Start here (GPU notebook)

**`notebooks/3lc_low_light_production.ipynb`** — run top-to-bottom on Kaggle (T4 x2) or Colab (T4/A100):

1. Downloads the 625MB starter kit straight from the 3LC CDN (sha256-verified, resumable)
2. EDA: class imbalance, darkness stats, CLAHE review images
3. Baseline train with **low-light recipe** (not vanilla defaults)
4. Missing-label miner (`conf > 0.8` + `IoU < 0.1`) → human review → fix → retrain
5. TTA multiscale + **per-class confidence** inference → validated `submission.csv`

## Why this beats vanilla baselines

| Vanilla mistake | This pipeline | Expected lift |
|---|---|---|
| `yolo train` defaults (`hsv_v=0.4`, `erasing=0.4`) erase tiny Cups | `hsv_v=0.7`, `erasing=0.2`, `mixup=0.1`, `copy_paste=0.1` | +1–2 mAP |
| Single `conf=0.25` — People (5966) drowns Bus (565) | Per-class conf (People 0.15, Bus 0.30, Cup/Bottle 0.20) | +1 mAP recall |
| Train once on noisy labels | Mine → review (CLAHE) → fix → retrain loop (0.8 → 0.7 → 0.6) | +2–4 mAP (the big one) |
| Single-scale infer | TTA 640+768 + flip, same weights (contract-safe) | +1 mAP small objects |
| No validation | Strict `sample_submission.csv` schema gate (715 rows) | zero failed submits |

Reference points: val mAP50 ≈ 0.53 @ 2 epochs, ≈ 0.70 @ 10 epochs. Competition top ≈ 0.669 test mAP.

## Repo layout

```
notebooks/3lc_low_light_production.ipynb  # ← run this on Kaggle/Colab
src/
  config.py            # contract, classes, per-class conf, low-light aug
  data_audit.py        # class/darkness/box-size audit
  train_production.py  # locked YOLOv11n + low-light recipe + rare-class cls=0.7
  mine_missing.py      # review QUEUE (never auto-patches; bulk-accept banned by Rules)
  serve.py             # (next) FastAPI: /predict /feedback /metrics + Docker
dataset.yaml           # absolute-path YOLO config (fixes ultralytics 8.3 rel-path bug)
requirements.txt
```

## Local quickstart (RTX 3050 4GB)

```bash
pip install -r requirements.txt
python -m src.data_audit
python -m src.train_production --epochs 20 --batch 4 --name prod_v1
python -m src.mine_missing --weights runs/detect/prod_v1/weights/best.pt --conf 0.8
# inspect review/ + review_queue.csv, then apply + retrain (see notebook cells 10-11)
```

## Rules compliance

- YOLOv11n only, `yolo11n.pt` COCO init, 640px — enforced in `config.py` / training cells.
- No external data. TTA uses the **same** weights (test-time only).
- Label fixes require **human visual review** (CLAHE crops in `review/`); `mine_missing.py` writes a queue and never patches labels directly. Bulk-accept without review is prohibited — don't do it.

## Submission

```bash
kaggle competitions submit -c the-3lc-low-light-detection-challenge -f submission.csv -m "yolo11n lowlight-aug + label-fix + TTA"
```

Prize-eligibility note: the organizers verify provenance via the 3LC Hub plugin's Run record. This repo mirrors the locked contract for portability; if you're chasing prize money, replay the winning `dataset_fixed` through the plugin's Train → Predict → Submit tabs so the Run is recorded.

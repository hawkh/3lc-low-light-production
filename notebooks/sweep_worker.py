"""One sweep run: train like the 3LC plugin, then score val like Kaggle.

Usage: python sweep_worker.py <config.json>
Config keys: name, out_dir, data_yaml, model, val_images, val_labels,
leak_stems, confs, deadline_ts (optional unix time), plugin (text),
train_args (ultralytics kwargs; only what the plugin accepts).
Writes <out_dir>/<name>/result.json. Skips if result.json exists.
"""
import json, sys, time
from pathlib import Path

LOCKED = {"imgsz": 640, "pretrained": True}
FORBIDDEN = {"model", "imgsz", "pretrained", "weights", "resume",
             "data", "tables", "settings", "project", "name", "exist_ok", "task"}
# Plugin field bounds (trainer._EXPOSED); the plugin checks extra args against them too.
BOUNDS = {"epochs": (1, 300), "batch": (1, 128), "lr0": (0.0001, 0.1),
          "lrf": (0.01, 1.0), "patience": (0, 100)}
PLUGIN_OPTIMIZERS = {"auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"}
NC = 12


def check_plugin_legal(args):
    bad = FORBIDDEN & set(args)
    if bad:
        raise ValueError(f"not settable in the plugin: {sorted(bad)}")
    for k, (lo, hi) in BOUNDS.items():
        if k in args and not lo <= args[k] <= hi:
            raise ValueError(f"{k}={args[k]} outside plugin bounds {lo}..{hi}")
    if args.get("optimizer", "auto") not in PLUGIN_OPTIMIZERS:
        raise ValueError(f"optimizer {args['optimizer']} is not in the plugin dropdown")


def load_val(val_images, val_labels):
    """[(stem, path, w, h, [(cls, x, y, w, h) in pixels])] for every val image."""
    from PIL import Image
    out = []
    for p in sorted(Path(val_images).glob("*.jpg")):
        w, h = Image.open(p).size
        gts = []
        lp = Path(val_labels) / f"{p.stem}.txt"
        if lp.exists():
            for line in lp.read_text().split("\n"):
                t = line.split()
                if len(t) >= 5:
                    c, xc, yc, bw, bh = int(float(t[0])), *map(float, t[1:5])
                    gts.append((c, (xc - bw / 2) * w, (yc - bh / 2) * h, bw * w, bh * h))
        out.append((p.stem, p, w, h, gts))
    return out


def predict_all(model, val, conf_min):
    """Plugin-identical inference (imgsz 640, max_det 300, default NMS IoU), once.

    Raising conf afterwards equals predicting at that conf: greedy NMS only lets a
    box be suppressed by a higher-scoring one, and the top-300 cap keeps by score.
    """
    preds = {}
    for i in range(0, len(val), 32):  # plugin predicts in chunks of 16-32 images
        chunk = val[i:i + 32]
        for (stem, _, w, h, _), r in zip(chunk, model.predict([str(v[1]) for v in chunk], imgsz=640,
                                                               conf=conf_min, max_det=300, verbose=False)):
            dets = []
            b = r.boxes
            if b is not None and len(b):
                for c, s, (xc, yc, bw, bh) in zip(b.cls.tolist(), b.conf.tolist(), b.xywhn.tolist()):
                    xc, yc, bw, bh = (min(1.0, max(0.0, v)) for v in (xc, yc, bw, bh))  # plugin clamps
                    dets.append((int(c), float(s), [(xc - bw / 2) * w, (yc - bh / 2) * h, bw * w, bh * h]))
            preds[stem] = dets
    return preds


def kaggle_ap50(val, preds, conf, exclude=frozenset()):
    """pycocotools bbox AP@0.5, maxDets=100, mean over classes that have GT."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import contextlib, io

    images, anns, dets = [], [], []
    for idx, (stem, _, w, h, gts) in enumerate((v for v in val if v[0] not in exclude), 1):
        images.append({"id": idx, "width": w, "height": h})
        for c, x, y, bw, bh in gts:
            anns.append({"id": len(anns) + 1, "image_id": idx, "category_id": c + 1, "iscrowd": 0,
                         "bbox": [x, y, bw, bh], "area": bw * bh})
        dets += [{"image_id": idx, "category_id": c + 1, "score": s, "bbox": bb}
                 for c, s, bb in preds[stem] if s >= conf]
    if not dets:
        return 0.0, {}
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns,
                  "categories": [{"id": k + 1} for k in range(NC)]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        ev = COCOeval(gt, gt.loadRes(dets), "bbox")
        ev.params.iouThrs = [0.5]
        ev.params.maxDets = [100]
        ev.params.areaRng = [ev.params.areaRng[0]]
        ev.params.areaRngLbl = ["all"]
        ev.evaluate(); ev.accumulate()
    prec = ev.eval["precision"][0, :, :, 0, 0]  # [recall, class]; -1 = class has no GT
    per_class = {k: float(prec[:, k][prec[:, k] > -1].mean()) for k in range(NC) if (prec[:, k] > -1).any()}
    return sum(per_class.values()) / len(per_class), per_class


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    out = Path(cfg["out_dir"]) / cfg["name"]
    res_path = out / "result.json"
    if res_path.exists():
        print(f"[{cfg['name']}] already done, skipping")
        return
    args = dict(cfg["train_args"])
    check_plugin_legal(args)

    from ultralytics import YOLO
    import pandas as pd

    deadline = cfg.get("deadline_ts")
    hit_deadline = {"v": False}

    def stop_at_deadline(trainer):
        if deadline and time.time() > deadline:
            hit_deadline["v"] = True
            trainer.stop = True  # ends after this epoch; best.pt/last.pt are still written

    t0 = time.time()
    model = YOLO(cfg["model"])
    model.add_callback("on_fit_epoch_end", stop_at_deadline)
    model.train(data=cfg["data_yaml"], project=str(Path(cfg["out_dir"])), name=cfg["name"],
                exist_ok=True, plots=False, **args, **LOCKED)
    train_s = time.time() - t0

    hist = pd.read_csv(out / "results.csv")
    hist.columns = [c.strip() for c in hist.columns]
    m50, m5095 = hist["metrics/mAP50(B)"], hist["metrics/mAP50-95(B)"]
    ep = lambda i: int(hist["epoch"][i]) + (1 if hist["epoch"].min() == 0 else 0)
    result = {
        "name": cfg["name"], "plugin": cfg.get("plugin", ""), "train_args": args,
        "epochs_run": int(len(hist)), "stopped_by_deadline": hit_deadline["v"],
        "train_minutes": round(train_s / 60, 1),
        "ul_best_map50": round(float(m50.max()), 4), "ul_best_map50_epoch": ep(m50.idxmax()),
        # ultralytics 8.4 saves best.pt by mAP50-95 only; Kaggle scores mAP50
        "ul_bestpt_epoch": ep(m5095.idxmax()), "ul_map50_at_bestpt": round(float(m50[m5095.idxmax()]), 4),
        "ul_last_map50": round(float(m50.iloc[-1]), 4),
        "map50_curve": [round(float(v), 4) for v in m50],
        "kaggle": {},
    }
    val = load_val(cfg["val_images"], cfg["val_labels"])
    leak = frozenset(cfg.get("leak_stems", []))
    for tag in ("best", "last"):
        w = out / "weights" / f"{tag}.pt"
        if not w.exists():
            continue
        preds = predict_all(YOLO(str(w)), val, min(cfg["confs"]))
        for conf in cfg["confs"]:
            full, per_class = kaggle_ap50(val, preds, conf)
            noleak, _ = kaggle_ap50(val, preds, conf, leak)
            result["kaggle"][f"{tag}@{conf}"] = {"ap50": round(full, 4), "ap50_noleak": round(noleak, 4),
                                                 "per_class": {k: round(v, 3) for k, v in per_class.items()}}
    res_path.write_text(json.dumps(result, indent=1))
    print(f"[{cfg['name']}] done in {train_s / 60:.1f} min")


if __name__ == "__main__":
    main()

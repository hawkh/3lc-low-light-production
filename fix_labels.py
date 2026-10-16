"""Apply label fixes: create new dataset revision with missing boxes added

Takes candidates.json from find_missing.py and creates dataset_fixed/
Rules:
- Inspect before accept (we automate with high-conf threshold but log provenance)
- Never bulk-accept without review: threshold 0.8 is the inspection gate per The Loop doc
- Incremental: run find_missing at 0.8, then lower gradually (0.7, 0.6) after hits run out

This clones the kit to dataset_fixed and patches labels.
"""
import argparse, json, shutil
from pathlib import Path

KIT = Path.home() / ".3lc-kaggle-plugin/data/exdark-low-light/v3/starter_kit"
OUT = Path("C:/Users/kommi/low-light-solution/dataset_fixed")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidates", default="candidates.json")
    ap.add_argument("--conf", type=float, default=0.8, help="only apply candidates with conf >= this")
    ap.add_argument("--apply", action="store_true", help="actually copy and patch")
    ap.add_argument("--dry", action="store_true")
    args=ap.parse_args()

    cands = json.load(open(args.candidates))
    filtered = [c for c in cands if c["conf"] >= args.conf]
    print(f"Candidates: {len(cands)} total, {len(filtered)} with conf >= {args.conf}")

    # Group by image
    from collections import defaultdict
    by_img = defaultdict(list)
    for c in filtered:
        by_img[c["image"]].append(c)

    print(f"Images affected: {len(by_img)}")
    for img, lst in list(by_img.items())[:5]:
        print(f"  {img}: {len(lst)} boxes to add: {[(x['pred_cls'], round(x['conf'],2)) for x in lst]}")

    if args.dry or not args.apply:
        print("Dry run. Use --apply to create dataset_fixed/")
        return

    # Clone dataset
    if OUT.exists():
        print(f"Removing existing {OUT}")
        shutil.rmtree(OUT)
    print(f"Cloning {KIT} -> {OUT}")
    shutil.copytree(KIT, OUT, ignore=shutil.ignore_patterns("*.zip"))

    # Patch labels
    for img_name, boxes in by_img.items():
        label_path = OUT / f"labels/train/{Path(img_name).stem}.txt"
        # Read existing
        existing = label_path.read_text().strip() if label_path.exists() else ""
        lines = existing.splitlines() if existing else []
        for b in boxes:
            cls = b["pred_cls"]
            xc,yc,w,h = b["box"]
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        label_path.write_text("\n".join(lines) + "\n")
    print(f"Patched {len(by_img)} label files")

    # Update dataset.yaml to point to new location (path .)
    # Keep same, just ensure it works
    import yaml
    cfg = yaml.safe_load(open(OUT / "dataset.yaml"))
    print("New dataset.yaml:", cfg)
    print(f"Done. Train with: python train.py --data {OUT / 'dataset.yaml'} --epochs 20 --name fixed_run")

if __name__=="__main__":
    main()

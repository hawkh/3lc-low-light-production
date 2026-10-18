"""Apply human-reviewed label fixes: review queue -> new dataset copy.

Only entries whose "status" a reviewer set to "accepted" are written.
Entries still "pending" are never applied (bulk-accept is banned by Rules).
The source dataset is copied, never edited in place.

Usage:
  python -m src.apply_fixes --queue review_queue.json            # dry run
  python -m src.apply_fixes --queue review_queue.json --apply
  python -m src.train_production --data dataset_fixed/dataset.yaml --name prod_v2_fixed
"""
import argparse, json, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path
from .config import KIT, CLASS_NAMES, write_dataset_yaml

STATUSES = {"pending", "accepted", "rejected"}


def validate(entry, i):
    """Return an error string for a malformed queue entry, else None."""
    if entry.get("status") not in STATUSES:
        return f"#{i}: status must be one of {sorted(STATUSES)}"
    if entry.get("cls") not in CLASS_NAMES:
        return f"#{i}: cls {entry.get('cls')!r} is not a class id"
    box = entry.get("box")
    if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)):
        return f"#{i}: box must be [xc, yc, w, h]"
    xc, yc, w, h = box
    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
        return f"#{i}: box {box} is not normalized"
    if not entry.get("image") or Path(entry["image"]).name != entry["image"]:
        return f"#{i}: image must be a bare file name"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="review_queue.json")
    ap.add_argument("--src", type=Path, default=KIT, help="dataset to copy (kit or previous fixed copy)")
    ap.add_argument("--out", type=Path, default=Path("dataset_fixed"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--apply", action="store_true", help="write the copy (default: dry run)")
    args = ap.parse_args()

    queue = json.loads(Path(args.queue).read_text())
    errors = [e for i, q in enumerate(queue) if (e := validate(q, i))]
    if errors:
        sys.exit("invalid queue entries:\n  " + "\n  ".join(errors[:20]))

    counts = Counter(q["status"] for q in queue)
    print(f"queue: {len(queue)} entries {dict(counts)}")
    by_img = defaultdict(list)
    for q in queue:
        if q["status"] == "accepted":
            by_img[q["image"]].append(q)
    if not by_img:
        sys.exit("no accepted entries: review the queue and set status to 'accepted' first")

    missing = [img for img in by_img if not (args.src / f"images/{args.split}/{img}").exists()]
    if missing:
        sys.exit(f"{len(missing)} accepted images not in {args.src}/images/{args.split}: {missing[:5]}")

    print(f"accepted: {counts['accepted']} boxes in {len(by_img)} images")
    if not args.apply:
        print("dry run. Add --apply to write", args.out)
        return

    src, out = args.src.resolve(), args.out.resolve()
    if out == src or src in out.parents or out in src.parents:
        sys.exit(f"--out {out} must not overlap --src {src}")
    if out.exists():
        sys.exit(f"{out} exists; remove it or pick another --out")

    shutil.copytree(src, out, ignore=shutil.ignore_patterns("*.zip"))
    for img, boxes in by_img.items():
        lp = out / f"labels/{args.split}/{Path(img).stem}.txt"
        lp.parent.mkdir(parents=True, exist_ok=True)
        lines = lp.read_text().strip().splitlines() if lp.exists() else []
        lines += ["%d %.6f %.6f %.6f %.6f" % (b["cls"], *b["box"]) for b in boxes]
        lp.write_text("\n".join(lines) + "\n")
    yml = write_dataset_yaml(out, out / "dataset.yaml")
    print(f"patched {len(by_img)} label files -> {out}")
    print(f"NEXT: python -m src.train_production --data {yml} --name prod_v2_fixed")


if __name__ == "__main__":
    main()

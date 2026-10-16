"""Optional low-light enhancement for visual inspection (not for model input)

Public papers (Zero-DCE, LLFlow) show enhancement helps human annotators spot
missed boxes in dark frames. Use this to brighten images when reviewing candidates.json
before accepting - per competition, accepting after visual inspection is allowed.

We apply CLAHE as cheap, model-agnostic enhancement for the review UI.
"""
import cv2
from pathlib import Path

def enhance_clahe(img_path, out_path, clip=2.0, grid=8):
    img = cv2.imread(str(img_path))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid,grid))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2,a,b))
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    cv2.imwrite(str(out_path), out)
    return out_path

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--out", default="enhanced.jpg")
    args=ap.parse_args()
    enhance_clahe(args.img, args.out)
    print(f"Wrote {args.out}")

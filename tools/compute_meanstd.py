#!/usr/bin/env python3
"""Compute dataset mean/std compatible with the pipeline's raw-intensity normalization.

The project's ToTensor transform does **not** rescale pixel intensities to [0, 1];
it simply casts the numpy array to float. Therefore, mean/std should typically be
computed on the original 8-bit values (0-255). Use the optional flag to rescale
if you intentionally want [0, 1] statistics.
"""
import argparse
import os
import sys
from typing import Iterable

import imageio.v2 as imageio
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(img_dir: str) -> Iterable[str]:
    for name in sorted(os.listdir(img_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMG_EXTS:
            yield os.path.join(img_dir, name)


def load_image(path: str, scale01: bool) -> np.ndarray:
    arr = imageio.imread(path)
    if arr.ndim == 2:  # grayscale HxW -> 1xHxW
        arr = arr[None, ...]
    elif arr.ndim == 3:  # HxWxC -> CxHxW
        arr = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported image dims {arr.shape} for {path}")

    arr = arr.astype(np.float64, copy=False)
    if scale01:
        arr = arr / 255.0
    return arr


def compute_stats(img_dir: str, scale01: bool):
    sums = None
    sumsq = None
    count = 0

    for path in list_images(img_dir):
        img = load_image(path, scale01)
        c, h, w = img.shape
        pixels = h * w
        if sums is None:
            sums = np.zeros(c, dtype=np.float64)
            sumsq = np.zeros(c, dtype=np.float64)
        elif sums.shape[0] != c:
            raise ValueError(
                f"Channel mismatch: expected {sums.shape[0]}, got {c} in {path}"
            )

        flat = img.reshape(c, pixels)
        sums += flat.sum(axis=1)
        sumsq += (flat ** 2).sum(axis=1)
        count += pixels

    if count == 0:
        raise RuntimeError(f"No images found in {img_dir}")

    mean = sums / count
    std = np.sqrt(sumsq / count - mean ** 2)
    return mean, std


def write_stats(mean: np.ndarray, std: np.ndarray, output: str):
    with open(output, "w", encoding="utf-8") as f:
        f.write("mean " + " ".join(f"{m:.4f}" for m in mean) + "\n")
        f.write("std " + " ".join(f"{s:.4f}" for s in std) + "\n")



def main():
    parser = argparse.ArgumentParser(description="Compute mean/std for mammogram images")
    parser.add_argument("--img_dir", required=True, help="Directory containing images (e.g., data/img)")
    parser.add_argument(
        "--output", default="data/meanstd.txt", help="Where to save the mean/std file"
    )
    parser.add_argument(
        "--scale01",
        action="store_true",
        help="Divide pixel values by 255.0 before computing statistics (not used in default pipeline)",
    )
    args = parser.parse_args()

    mean, std = compute_stats(args.img_dir, args.scale01)
    write_stats(mean, std, args.output)

    print(f"Processed {args.img_dir}")
    print("mean:", " ".join(f"{m:.4f}" for m in mean))
    print("std:", " ".join(f"{s:.4f}" for s in std))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    sys.exit(main())

"""Run LaMa inpainting on an image + mask pair.

Requires:  pip install simple-lama-inpainting

Mask convention (snapshot.py output):
  white (255) = good / keep,  black (0) = needs inpainting.
This is automatically inverted to LaMa's convention before inference.
Pass --no-invert if your mask already uses LaMa convention (white = inpaint).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


def main():
    parser = argparse.ArgumentParser(
        description="Inpaint an image using LaMa and a quality mask.",
    )
    parser.add_argument("image", type=Path, help="Input image (png/jpg).")
    parser.add_argument("mask", type=Path, help="Mask image (grayscale png).")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output path (default: <image_stem>_inpainted.png).")
    parser.add_argument("--no-invert", action="store_true",
                        help="Skip mask inversion (use if mask is already LaMa "
                             "convention: white=inpaint, black=keep).")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: cpu, mps, cuda (default: auto-detect).")
    args = parser.parse_args()

    # Resolve output path.
    output = args.output or args.image.with_name(
        f"{args.image.stem}_inpainted{args.image.suffix}"
    )

    # Load inputs.
    image = Image.open(args.image).convert("RGB")
    mask = Image.open(args.mask).convert("L")

    if image.size != mask.size:
        print(f"Warning: resizing mask {mask.size} to match image {image.size}")
        mask = mask.resize(image.size, Image.NEAREST)

    # Invert mask: snapshot.py outputs white=good, LaMa expects white=inpaint.
    if not args.no_invert:
        mask = ImageOps.invert(mask)

    import torch
    from simple_lama_inpainting import SimpleLama

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading LaMa model (device={device})...")
    lama = SimpleLama(device=device)

    print(f"Inpainting {args.image} ...")
    result = lama(image, mask)

    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(output))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

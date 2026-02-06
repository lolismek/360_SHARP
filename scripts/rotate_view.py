#!/usr/bin/env python3
"""End-to-end: image -> SHARP prediction -> rotated snapshot.

Runs SHARP Gaussian prediction on an input image, then renders
a snapshot from a rotated viewpoint at the original resolution.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Predict Gaussians from an image and render a rotated snapshot.",
    )
    parser.add_argument("input", type=Path, help="Input image (jpg/png).")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output snapshot path (default: <input_stem>_rotated.png).")
    parser.add_argument("-a", "--angle", type=float, default=5.0,
                        help="Yaw rotation in degrees (positive = right, default: 5).")
    parser.add_argument("--mask-mode", choices=["quality", "alpha"], default="quality",
                        help="Mask type passed to snapshot (default: quality).")
    parser.add_argument("--ply-dir", type=Path, default=None,
                        help="Directory for intermediate .ply (default: temp dir).")
    parser.add_argument("--keep-ply", action="store_true",
                        help="Keep the intermediate .ply file.")
    parser.add_argument("--device", type=str, default="default",
                        help="Device for SHARP prediction: cpu, mps, cuda, default.")
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        sys.exit(f"Error: {input_path} not found")

    if args.output is None:
        output_path = input_path.parent / f"{input_path.stem}_rotated.png"
    else:
        output_path = args.output.resolve()

    use_tmp = args.ply_dir is None and not args.keep_ply
    if args.ply_dir:
        ply_dir = args.ply_dir.resolve()
        ply_dir.mkdir(parents=True, exist_ok=True)
    elif args.keep_ply:
        ply_dir = output_path.parent
    else:
        ply_dir = Path(tempfile.mkdtemp(prefix="sharp_"))

    ply_path = ply_dir / f"{input_path.stem}.ply"

    try:
        # --- Step 1: SHARP prediction ---
        print(f"=== Predicting Gaussians from {input_path.name} ===")
        cmd = ["sharp", "predict", "-i", str(input_path), "-o", str(ply_dir)]
        if args.device != "default":
            cmd += ["--device", args.device]
        subprocess.run(cmd, check=True)

        if not ply_path.exists():
            sys.exit(f"Error: expected {ply_path} not found after prediction")

        # --- Step 2: Render rotated snapshot ---
        print(f"\n=== Rendering snapshot (angle={args.angle}deg) ===")
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "snapshot.py"),
            str(ply_path),
            "-a", str(args.angle),
            "-o", str(output_path),
            "--mask-mode", args.mask_mode,
        ]
        subprocess.run(cmd, check=True)

        print(f"\nDone: {output_path}")

    finally:
        if use_tmp and ply_dir.exists():
            shutil.rmtree(ply_dir)


if __name__ == "__main__":
    main()

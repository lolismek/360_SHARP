#!/usr/bin/env python3
"""Iterative orbit loop: predict -> rotate -> snapshot -> re-predict -> ...

Starting from an input image, repeatedly:
  1. Run SHARP prediction to get a .ply
  2. Render a snapshot rotated by --angle degrees from the current viewpoint
  3. Use that snapshot as input for the next iteration

All intermediate images and (optionally) .ply files are saved to --output-dir.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent


def predict(image_path: Path, ply_dir: Path, device: str) -> Path:
    """Run SHARP prediction, return path to the output .ply."""
    cmd = ["sharp", "predict", "-i", str(image_path), "-o", str(ply_dir)]
    if device != "default":
        cmd += ["--device", device]
    subprocess.run(cmd, check=True)
    ply_path = ply_dir / f"{image_path.stem}.ply"
    if not ply_path.exists():
        sys.exit(f"Error: expected {ply_path} not found after prediction")
    return ply_path


def render_snapshot(
    ply_path: Path,
    angle: float,
    output_path: Path,
    mask_mode: str = "quality",
) -> None:
    """Render a rotated snapshot from a .ply file."""
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "snapshot.py"),
        str(ply_path),
        "-a", str(angle),
        "-o", str(output_path),
        "--mask-mode", mask_mode,
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Iterative orbit: predict, rotate, snapshot, repeat.",
    )
    parser.add_argument("input", type=Path, help="Initial input image (jpg/png).")
    parser.add_argument("-o", "--output-dir", type=Path, required=True,
                        help="Directory to save all outputs.")
    parser.add_argument("-n", "--steps", type=int, default=5,
                        help="Number of orbit iterations (default: 5).")
    parser.add_argument("-a", "--angle", type=float, default=5.0,
                        help="Rotation per step in degrees (default: 5).")
    parser.add_argument("--mask-mode", choices=["quality", "alpha"], default="quality",
                        help="Mask type passed to snapshot (default: quality).")
    parser.add_argument("--keep-ply", action="store_true",
                        help="Keep intermediate .ply files.")
    parser.add_argument("--device", type=str, default="default",
                        help="Device for SHARP prediction: cpu, mps, cuda, default.")
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        sys.exit(f"Error: {input_path} not found")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy the original image into the output directory as step 0.
    step0_path = output_dir / "step_00.png"
    shutil.copy2(input_path, step0_path)
    print(f"Step 0: {input_path.name} -> {step0_path.name}")

    current_image = input_path

    for i in range(1, args.steps + 1):
        print(f"\n{'='*60}")
        print(f"Step {i}/{args.steps}  (cumulative rotation: {i * args.angle:.1f} deg)")
        print(f"{'='*60}")

        # --- Predict ---
        if args.keep_ply:
            ply_dir = output_dir
        else:
            ply_dir = Path(tempfile.mkdtemp(prefix="sharp_"))

        try:
            print(f"\n--- Predicting from {current_image.name} ---")
            ply_path = predict(current_image, ply_dir, args.device)

            # --- Render rotated snapshot ---
            snapshot_path = output_dir / f"step_{i:02d}.png"
            print(f"\n--- Rendering snapshot (angle={args.angle} deg) ---")
            render_snapshot(ply_path, args.angle, snapshot_path, args.mask_mode)

            print(f"\nSaved {snapshot_path.name}")
            current_image = snapshot_path

        finally:
            if not args.keep_ply and ply_dir != output_dir and ply_dir.exists():
                shutil.rmtree(ply_dir)

    print(f"\n{'='*60}")
    print(f"Done: {args.steps} steps, {args.steps * args.angle:.1f} deg total rotation")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

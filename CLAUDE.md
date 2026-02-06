# CLAUDE.md

## Project overview

360-SHARP generates 360-degree views from a single photo. It wraps Apple's SHARP model with scripts for orbit rendering, inpainting, and web viewing.

## Build & run

```bash
# Install (from ml-sharp/)
conda activate sharp
pip install -r ml-sharp/requirements.txt

# Predict Gaussians
sharp predict -i images/test_sharp.jpg -o results/

# Render rotated snapshot (CPU)
python scripts/snapshot.py results/test_sharp.ply -a 10 -o results/orbit_10.png

# Inpaint (requires: pip install simple-lama-inpainting)
python scripts/inpaint.py results/orbit_10.png results/orbit_10_mask.png -o results/orbit_10_fixed.png

# Iterative orbit
python scripts/orbit_loop.py images/test_sharp.jpg -o results/orbit/ -n 5 -a 5

# Convert for web viewer
python splat-viewer/convert.py results/test_sharp.ply -o splat-viewer/model.splat
```

## Architecture

### ml-sharp/src/sharp/ (Apple's SHARP package)

- `cli/` -- Click CLI entry points. `predict.py` runs inference, `render.py` renders trajectory videos (CUDA only).
- `models/predictor.py` -- Main `RGBGaussianPredictor`: image -> 3D Gaussians via ViT encoder + DPT depth + Gaussian decoder + composer.
- `models/params.py` -- All hyperparameters as dataclasses (`PredictorParams`, `MonodepthParams`, etc).
- `utils/gaussians.py` -- `Gaussians3D` dataclass, PLY I/O (`save_ply`, `load_ply`), `unproject_gaussians`.
- `utils/camera.py` -- Camera models and trajectory generation.
- `utils/gsplat.py` -- gsplat CUDA renderer wrapper.

### scripts/ (pipeline scripts)

- `snapshot.py` -- CPU software rasterizer for rendering rotated views from a PLY. Supports `--mask-mode quality` (coverage * sharpness + morphology) and `--mask-mode alpha` (raw coverage).
- `orbit_loop.py` -- Iterative predict -> rotate -> re-predict loop.
- `rotate_view.py` -- Single predict + snapshot in one command.
- `inpaint.py` -- LaMa inpainting with mask from snapshot.

### splat-viewer/ (web viewer)

- `convert.py` -- PLY -> .splat binary format.
- `main.js` + `index.html` -- WebGL Gaussian Splat viewer (no deps).

## Key patterns

- Coordinate convention: OpenCV (X-right, Y-down, Z-forward). Scene center at (0, 0, +z).
- Color space: model outputs linearRGB; scripts convert to sRGB (gamma 2.2) before saving PNGs.
- Mask convention: white = good/keep, black = needs inpainting. `inpaint.py` auto-inverts for LaMa.
- Scripts invoke each other via subprocess (not direct imports) to keep them standalone.
- Model checkpoint (~250 MB) auto-downloads to `~/.cache/torch/hub/checkpoints/` on first run.

## Linting

```bash
cd ml-sharp
ruff check src/
```

Uses ruff with line-length 100, Google docstring convention. Config in `ml-sharp/pyproject.toml`.

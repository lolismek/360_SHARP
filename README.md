# 360-SHARP

Generate 360-degree views from a single photograph using [SHARP](https://arxiv.org/abs/2512.10685) (Sharp Monocular View Synthesis).

This project wraps Apple's [ml-sharp](https://github.com/apple/ml-sharp) model with a pipeline for iterative orbit rendering, inpainting, and interactive 3D viewing. Given one image, it predicts a 3D Gaussian Splat, renders rotated snapshots, fills in missing regions with LaMa inpainting, and optionally serves the result in a WebGL viewer.

## Pipeline overview

```
Input image (.jpg/.png)
    |
    v
[1] sharp predict          -- Single-image -> 3D Gaussian Splat (.ply)
    |
    v
[2] snapshot.py            -- Render a rotated view + quality mask (CPU)
    |
    v
[3] inpaint.py             -- Fill masked regions with LaMa
    |
    v
[4] orbit_loop.py          -- Repeat steps 1-3 to sweep a full orbit
    |
    v
[5] convert.py + viewer    -- Convert .ply -> .splat, view in browser
```

## Setup

```bash
# Create environment
conda create -n sharp python=3.13
conda activate sharp

# Install SHARP and dependencies
cd ml-sharp
pip install -r requirements.txt

# (Optional) Install LaMa for inpainting
pip install simple-lama-inpainting
```

The model checkpoint (~250 MB) downloads automatically on first run.

## Usage

### 1. Predict Gaussians from an image

```bash
sharp predict -i images/test_sharp.jpg -o results/
# Output: results/test_sharp.ply
```

### 2. Render a rotated snapshot (CPU, no CUDA needed)

```bash
# Render at 10-degree orbit with quality mask
python scripts/snapshot.py results/test_sharp.ply \
    -a 10 -o results/orbit_10.png

# Use alpha-only mask (simpler, no sharpness weighting)
python scripts/snapshot.py results/test_sharp.ply \
    -a 10 -o results/orbit_10.png --mask-mode alpha
```

### 3. Inpaint missing regions

```bash
python scripts/inpaint.py results/orbit_10.png results/orbit_10_mask.png \
    -o results/orbit_10_fixed.png
```

### 4. End-to-end: predict + rotate in one command

```bash
python scripts/rotate_view.py images/test_sharp.jpg -a 10
```

### 5. Iterative orbit loop (multi-step rotation)

```bash
python scripts/orbit_loop.py images/test_sharp.jpg \
    -o results/orbit/ -n 5 -a 5
# Produces: step_00.png through step_05.png (25 deg total)
```

### 6. Render trajectory video (CUDA only)

```bash
sharp render -i results/test_sharp.ply -o results/
```

### 7. View in browser

```bash
# Convert PLY to web-friendly .splat format
python splat-viewer/convert.py results/test_sharp.ply -o splat-viewer/model.splat

# Open splat-viewer/index.html in a browser
# Or serve locally: python -m http.server -d splat-viewer 8000
```

## Project structure

```
360-SHARP/
├── README.md
├── CLAUDE.md                       # Dev guidance for Claude Code
├── scripts/                        # Pipeline scripts
│   ├── snapshot.py                 # Render snapshot from PLY (CPU)
│   ├── orbit_loop.py               # Iterative predict-rotate loop
│   ├── rotate_view.py              # Single predict + rotate
│   └── inpaint.py                  # LaMa inpainting with mask
├── ml-sharp/                       # SHARP model (Apple ML Research)
│   ├── src/sharp/
│   │   ├── cli/                    # CLI entry points (predict, render)
│   │   ├── models/                 # Neural network architecture
│   │   │   ├── predictor.py        # Main model: RGBGaussianPredictor
│   │   │   ├── monodepth.py        # Dense Prediction Transformer
│   │   │   ├── gaussian_decoder.py # Gaussian parameter decoder
│   │   │   ├── composer.py         # Base + delta Gaussian composer
│   │   │   ├── encoders/           # ViT, SPN, UNet encoders
│   │   │   └── decoders/           # Multi-resolution decoders
│   │   └── utils/                  # Gaussians, camera, I/O, rendering
│   ├── pyproject.toml
│   └── README.md                   # Apple's original documentation
├── splat-viewer/                   # WebGL Gaussian Splat viewer
│   ├── main.js                     # WebGL renderer (no dependencies)
│   ├── index.html                  # Viewer page
│   ├── convert.py                  # PLY -> .splat converter
│   └── README.md
└── images/                         # Input images
    └── test_sharp.jpg
```

## Key concepts

- **Gaussian Splatting**: Represents 3D scenes as collections of oriented, colored, translucent 3D Gaussians. Each Gaussian has a position, scale, rotation (quaternion), color, and opacity.
- **PLY format**: Standard point cloud format used to store Gaussian parameters. SHARP uses OpenCV coordinate convention (X-right, Y-down, Z-forward).
- **Quality mask**: After rendering a rotated view, regions with poor coverage or large (blurry) splats are masked. These regions can then be filled by inpainting.
- **Software rasterizer**: The `snapshot.py` script includes a pure-PyTorch rasterizer that works on CPU/MPS without CUDA, at the cost of speed.

## Dependencies

| Package | Purpose |
|---------|---------|
| torch, torchvision | Deep learning framework |
| timm | Vision Transformer models |
| gsplat | CUDA Gaussian splatting (for `sharp render`) |
| imageio | Image/video I/O |
| plyfile | PLY format read/write |
| click | CLI framework |
| simple-lama-inpainting | Inpainting (optional) |

## Credits

- **SHARP model**: [Apple ML Research](https://github.com/apple/ml-sharp) -- Mescheder et al., "Sharp Monocular View Synthesis in Less Than a Second" ([arXiv:2512.10685](https://arxiv.org/abs/2512.10685))
- **WebGL viewer**: [antimatter15/splat](https://github.com/antimatter15/splat) by Kevin Kwok

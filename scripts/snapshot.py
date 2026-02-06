"""Render a snapshot of a Gaussian splat PLY, optionally rotated.

Works on CPU and MPS (no CUDA required). Uses a pure-PyTorch software
rasterizer as fallback when gsplat CUDA kernels are unavailable.

Supports two mask modes:
  --mask-mode quality  (default) Combined alpha-coverage * scale-based
                       sharpness, with blur + morphological closing.
  --mask-mode alpha    Raw transmittance mask (1 - T) with no post-processing.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as iio

from sharp.utils.gaussians import load_ply


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def make_orbit_extrinsics(
    angle_deg: float, orbit_center: torch.Tensor, device: torch.device,
) -> torch.Tensor:
    """4x4 extrinsics that orbit the camera around orbit_center on the Y axis.

    Positive angle = orbit right (camera moves right, subject seen from its right).
    The camera always faces the orbit center.
    """
    # Negate so positive angle = camera moves in +X direction (right).
    theta = -math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)

    cx = orbit_center[0].item()
    cy = orbit_center[1].item()
    cz = orbit_center[2].item()

    # Camera starts at origin. Rotate (origin − center) around Y, add center back.
    new_x = cx * (1 - c) - cz * s
    new_y = 0.0  # keep camera at original height
    new_z = cz * (1 - c) + cx * s

    # --- look-at (OpenCV: X-right, Y-down, Z-forward) ---
    fwd = torch.tensor([cx - new_x, cy - new_y, cz - new_z],
                        device=device, dtype=torch.float32)
    fwd = fwd / fwd.norm()

    world_up = torch.tensor([0.0, -1.0, 0.0], device=device)  # -Y = up
    right = torch.linalg.cross(fwd, world_up)
    right = right / right.norm()
    down = torch.linalg.cross(fwd, right)

    R = torch.stack([right, down, fwd], dim=0)                 # (3, 3)
    P = torch.tensor([new_x, new_y, new_z], device=device, dtype=torch.float32)
    t = -R @ P

    E = torch.eye(4, device=device, dtype=torch.float32)
    E[:3, :3] = R
    E[:3, 3] = t
    return E


# ---------------------------------------------------------------------------
# Pure-PyTorch software Gaussian rasterizer
# ---------------------------------------------------------------------------

def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(N, 4) wxyz quaternions -> (N, 3, 3) rotation matrices."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
        2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)


@torch.no_grad()
def software_render(
    means: torch.Tensor,       # (N, 3)
    quats: torch.Tensor,       # (N, 4)
    scales: torch.Tensor,      # (N, 3) already exponentiated singular values
    colors: torch.Tensor,      # (N, 3)
    opacities: torch.Tensor,   # (N,)
    extrinsics: torch.Tensor,  # (4, 4)
    intrinsics: torch.Tensor,  # (4, 4)
    W: int,
    H: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render Gaussians to an (H, W, 3) image using software rasterization.

    Returns (image, alpha_mask, avg_scale) where:
      - alpha_mask: per-pixel coverage (1 = covered, 0 = empty)
      - avg_scale:  weighted-average 2D splat radius per pixel (in pixels)
    """
    device = means.device
    R_cam = extrinsics[:3, :3]
    t_cam = extrinsics[:3, 3]

    # --- 1. World → camera transform (vectorised) ---
    means_cam = means @ R_cam.T + t_cam  # (N, 3)

    # Keep only Gaussians in front of the camera.
    visible = means_cam[:, 2] > 0.2
    means_cam = means_cam[visible]
    quats = quats[visible]
    scales = scales[visible]
    colors = colors[visible]
    opacities = opacities[visible]

    # --- 2. Project centres to 2-D ---
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    z = means_cam[:, 2]
    cen_x = fx * means_cam[:, 0] / z + cx
    cen_y = fy * means_cam[:, 1] / z + cy

    # Discard Gaussians clearly outside viewport.
    margin = 200
    in_view = (cen_x > -margin) & (cen_x < W + margin) & (cen_y > -margin) & (cen_y < H + margin)
    means_cam, cen_x, cen_y = means_cam[in_view], cen_x[in_view], cen_y[in_view]
    quats, scales, colors, opacities = quats[in_view], scales[in_view], colors[in_view], opacities[in_view]
    z = means_cam[:, 2]

    # --- 3. 2-D covariance ---
    R_g = _quat_to_rotmat(quats)                         # (N,3,3)
    S2 = torch.diag_embed(scales ** 2)                    # (N,3,3)
    cov3d = R_g @ S2 @ R_g.transpose(-1, -2)             # (N,3,3)

    # Jacobian of pinhole projection.
    J = torch.zeros(len(z), 2, 3, device=device)
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * means_cam[:, 0] / (z * z)
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * means_cam[:, 1] / (z * z)

    cov2d = J @ (R_cam @ cov3d @ R_cam.T) @ J.transpose(-1, -2)  # (N,2,2)
    # Low-pass filter (anti-aliasing).
    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3

    # Inverse of 2×2 symmetric covariance.
    a, b, c = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    det = a * c - b * b
    inv_det = 1.0 / det.clamp(min=1e-10)
    inv00 = c * inv_det
    inv11 = a * inv_det
    inv01 = -b * inv_det

    # Splat radius (3σ of the major axis).
    mid = 0.5 * (a + c)
    disc = torch.sqrt((mid * mid - det).clamp(min=0.0))
    radius = torch.ceil(3.0 * torch.sqrt((mid + disc).clamp(min=0.01))).int()
    radius = radius.clamp(max=128)

    # --- 4. Sort front-to-back ---
    order = torch.argsort(z)

    # Filter tiny / transparent Gaussians.
    keep = (radius[order] >= 1) & (opacities[order] > 0.005)
    order = order[keep]

    N_render = len(order)
    print(f"  Rendering {N_render} Gaussians (filtered from {visible.sum().item()} visible)")

    # --- 5. Front-to-back alpha compositing ---
    image = torch.zeros(H, W, 3, device=device)
    T = torch.ones(H, W, device=device)
    scale_sum = torch.zeros(H, W, device=device)
    weight_sum = torch.zeros(H, W, device=device)

    t0 = time.time()
    for step, idx in enumerate(order):
        if step % 50000 == 0:
            elapsed = time.time() - t0
            pct = 100.0 * step / max(N_render, 1)
            print(f"  [{pct:5.1f}%] {step}/{N_render}  ({elapsed:.1f}s)")

        r = radius[idx].item()
        mx = cen_x[idx].item()
        my = cen_y[idx].item()

        x0, x1 = max(0, int(mx - r)), min(W, int(mx + r + 1))
        y0, y1 = max(0, int(my - r)), min(H, int(my + r + 1))
        if x0 >= x1 or y0 >= y1:
            continue

        # Local pixel grid.
        px = torch.arange(x0, x1, device=device, dtype=torch.float32) - mx
        py = torch.arange(y0, y1, device=device, dtype=torch.float32) - my
        gy, gx = torch.meshgrid(py, px, indexing="ij")

        # Mahalanobis distance → Gaussian weight.
        maha = inv00[idx] * gx * gx + 2 * inv01[idx] * gx * gy + inv11[idx] * gy * gy
        gauss = torch.exp(-0.5 * maha)
        alpha = (opacities[idx] * gauss).clamp(max=0.99)

        # Skip negligible contribution.
        if alpha.max().item() < 1.0 / 255.0:
            continue

        T_tile = T[y0:y1, x0:x1]
        weight = alpha * T_tile
        image[y0:y1, x0:x1] += weight.unsqueeze(-1) * colors[idx]
        T[y0:y1, x0:x1] = T_tile * (1 - alpha)
        scale_sum[y0:y1, x0:x1] += weight * float(r)
        weight_sum[y0:y1, x0:x1] += weight

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    alpha_mask = 1 - T  # per-pixel coverage: 1 = fully covered, 0 = no coverage
    avg_scale = scale_sum / weight_sum.clamp(min=1e-6)
    return image.clamp(0, 1), alpha_mask, avg_scale


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def make_quality_mask(
    alpha_mask: torch.Tensor,
    avg_scale: torch.Tensor,
    scale_threshold: float = 10.0,
    mask_blur: int = 21,
    mask_threshold: float = 0.5,
    morph_size: int = 15,
) -> torch.Tensor:
    """Combined quality mask: alpha coverage * scale-based sharpness.

    sharpness ~ 1 when avg splat radius is small (sharp detail),
    drops toward 0 for large radii (blurry filler Gaussians).
    """
    sharpness = 1.0 / (1.0 + (avg_scale / scale_threshold).pow(2))
    quality = alpha_mask * sharpness

    # 1. Gaussian blur to merge nearby noisy spots into contiguous regions.
    blur = mask_blur | 1  # ensure odd
    if blur >= 3:
        sigma = blur / 4.0
        ax = torch.arange(blur, device=quality.device, dtype=torch.float32) - blur // 2
        kern_1d = torch.exp(-ax.pow(2) / (2 * sigma ** 2))
        kern_2d = kern_1d.outer(kern_1d)
        kern_2d = (kern_2d / kern_2d.sum()).unsqueeze(0).unsqueeze(0)  # (1,1,k,k)
        quality = F.conv2d(
            quality.unsqueeze(0).unsqueeze(0), kern_2d, padding=blur // 2,
        ).squeeze()

    # 2. Binary threshold.
    quality = (quality >= mask_threshold).float()

    # 3. Morphological close on the inpaint (dark) regions so they become
    #    contiguous blobs instead of scattered speckles.
    morph = morph_size | 1  # ensure odd
    if morph >= 3:
        pad = morph // 2
        bad = (1 - quality).unsqueeze(0).unsqueeze(0)        # (1,1,H,W)
        bad = F.max_pool2d(bad, morph, stride=1, padding=pad)  # dilate
        bad = -F.max_pool2d(-bad, morph, stride=1, padding=pad)  # erode
        quality = 1 - bad.squeeze()

    return quality


# ---------------------------------------------------------------------------
# Shared rendering pipeline
# ---------------------------------------------------------------------------

def render_snapshot(
    ply_path: Path,
    output_path: Path,
    angle: float = 0.0,
    mask_mode: str = "quality",
    mask_output: Path | None = None,
    scale_threshold: float = 10.0,
    mask_blur: int = 21,
    mask_threshold: float = 0.5,
    morph_size: int = 15,
) -> None:
    """Load a PLY, render at the given orbit angle, and save image + mask."""
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load scene.
    gaussians, metadata = load_ply(ply_path)
    width, height = metadata.resolution_px
    f_px = metadata.focal_length_px
    print(f"Loaded {ply_path}: {width}x{height}, f={f_px:.1f}px")

    # Build intrinsics.
    intrinsics = torch.tensor([
        [f_px, 0, (width - 1) / 2.0, 0],
        [0, f_px, (height - 1) / 2.0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], device=device, dtype=torch.float32)

    # Orbit around the center of the frame at the median scene depth.
    median_z = gaussians.mean_vectors[0][:, 2].median().item()
    orbit_center = torch.tensor([0.0, 0.0, median_z], device=device)
    print(f"Orbit center: (0, 0, {median_z:.2f})")

    # Build extrinsics: orbit camera around the subject.
    extrinsics = make_orbit_extrinsics(angle, orbit_center, device)

    # Render.
    image, alpha_mask, avg_scale = software_render(
        means=gaussians.mean_vectors[0].to(device),
        quats=gaussians.quaternions[0].to(device),
        scales=gaussians.singular_values[0].to(device),
        colors=gaussians.colors[0].to(device),
        opacities=gaussians.opacities[0].to(device),
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        W=width,
        H=height,
    )

    # Color-space: SHARP stores colors in linearRGB; convert to sRGB for saving.
    if metadata.color_space == "linearRGB":
        image = image.clamp(min=0.0).pow(1.0 / 2.2)

    # Save image.
    image_np = (image * 255).to(torch.uint8).numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(output_path), image_np)
    print(f"Saved {output_path} ({width}x{height}, angle={angle}\u00b0)")

    # Generate mask based on mode.
    if mask_mode == "quality":
        mask = make_quality_mask(
            alpha_mask, avg_scale,
            scale_threshold=scale_threshold,
            mask_blur=mask_blur,
            mask_threshold=mask_threshold,
            morph_size=morph_size,
        )
    else:
        # "alpha" mode: raw transmittance, no post-processing.
        mask = alpha_mask

    # Save mask.
    mask_path = mask_output or output_path.with_name(
        f"{output_path.stem}_mask{output_path.suffix}"
    )
    mask_np = (mask.clamp(0, 1) * 255).to(torch.uint8).numpy()
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(mask_path), mask_np)
    print(f"Saved mask {mask_path} (mode={mask_mode})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Snapshot a Gaussian splat PLY.")
    parser.add_argument("input", type=Path, help="Path to the .ply file.")
    parser.add_argument("-o", "--output", type=Path, default=Path("snapshot.png"),
                        help="Output image path (default: snapshot.png).")
    parser.add_argument("-a", "--angle", type=float, default=0.0,
                        help="Yaw rotation in degrees (positive = right).")
    parser.add_argument("--mask-mode", choices=["quality", "alpha"], default="quality",
                        help="Mask type: 'quality' (coverage * sharpness + morphology) "
                             "or 'alpha' (raw coverage only). Default: quality.")
    parser.add_argument("--mask-output", type=Path, default=None,
                        help="Output path for mask (default: <output_stem>_mask.png).")
    parser.add_argument("--scale-threshold", type=float, default=10.0,
                        help="Avg splat-radius (px) at which sharpness drops to 50%% "
                             "(quality mode only, default: 10).")
    parser.add_argument("--mask-blur", type=int, default=21,
                        help="Gaussian blur kernel size for mask smoothing, 0 to disable "
                             "(quality mode only, default: 21).")
    parser.add_argument("--mask-threshold", type=float, default=0.5,
                        help="Binarisation threshold for the quality mask "
                             "(quality mode only, default: 0.5).")
    parser.add_argument("--morph-size", type=int, default=15,
                        help="Morphological-close kernel size, 0 to disable "
                             "(quality mode only, default: 15).")
    args = parser.parse_args()

    render_snapshot(
        ply_path=args.input,
        output_path=args.output,
        angle=args.angle,
        mask_mode=args.mask_mode,
        mask_output=args.mask_output,
        scale_threshold=args.scale_threshold,
        mask_blur=args.mask_blur,
        mask_threshold=args.mask_threshold,
        morph_size=args.morph_size,
    )


if __name__ == "__main__":
    main()

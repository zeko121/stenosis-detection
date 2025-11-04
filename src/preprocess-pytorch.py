"""
CCTA Image Preprocessing Module (PyTorch/MONAI-style)

This module handles the preprocessing of Coronary CT Angiography (CCTA) scans and their associated
binary masks. Key operations include:
1. Loading and saving NIFTI format medical images
2. Resampling volumes to isotropic spacing (default 1mm³)
3. HU value clipping and normalization for CT images
4. Binary mask resampling with topology preservation
5. Geometric metadata handling (affine transformations)
6. Extensive validation and quality checks

The preprocessing pipeline maintains proper geometric relationships and validates all
transformations to ensure data integrity.

PyTorch/MONAI-style notes:
- Resampling is performed via torch.nn.functional.interpolate on 3D tensors
  (N, C, D, H, W), optionally on GPU if available.
- The public API (function names, signatures, and docstrings) is preserved.
- I/O remains nibabel-based for portability; you can switch to MONAI IO utilities later.
- No training code is included here; this module focuses strictly on preprocessing.
"""

from __future__ import annotations

import os
import csv
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

import time


# (Optional) MONAI import for consistency/style; not strictly required for this file's logic.
try:
    import monai  # noqa: F401
except Exception:  # pragma: no cover
    monai = None

# Global configuration
VERBOSE = True       # Enable detailed processing logs
RUN_CHECKS = True    # Perform validation checks during processing
ATOL = 1e-3          # Absolute tolerance for floating point comparisons
USE_GPU = torch.cuda.is_available()  # If True and CUDA exists, resampling will run on GPU

# ---------- Helpers (PyTorch resampling) ----------

def _compute_zoom_and_outshape(orig_spacing: np.ndarray,
                               new_spacing: np.ndarray,
                               in_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Compute zoom factors and output shape given spacings and input shape."""
    zoom = orig_spacing / new_spacing
    Dz, Hy, Wx = in_shape  # in numpy order (Z, Y, X)
    out_shape = tuple(int(round(s * z)) for s, z in zip(in_shape, zoom))
    return zoom, out_shape


def _torch_resample_3d(volume_np: np.ndarray,
                       out_shape: Tuple[int, int, int],
                       mode: str = "trilinear") -> np.ndarray:
    """Resample a 3D numpy array to ``out_shape`` using torch.interpolate.

    Args:
        volume_np: 3D array (Z, Y, X)
        out_shape: desired (Z, Y, X)
        mode: 'trilinear' for images, 'nearest' for masks

    Returns:
        Resampled array as float32 (or uint8 for masks if post-processed later)
    """
    assert volume_np.ndim == 3, "Expected a 3D array (Z,Y,X)."
    # torch expects (N, C, D, H, W)
    vol_t = torch.from_numpy(volume_np[None, None, ...])  # (1,1,D,H,W)
    device = torch.device("cuda" if USE_GPU else "cpu")
    vol_t = vol_t.to(device=device, dtype=torch.float32)

    # align_corners=False is the standard safe default for medical volumes
    outD, outH, outW = out_shape
    res = F.interpolate(vol_t, size=(outD, outH, outW), mode=mode, align_corners=False if mode != "nearest" else None)
    res = res[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
    return res


# ---------- Affine Transformation Utilities ----------

def spacing_from_affine(affine: np.ndarray) -> np.ndarray:
    """Extract voxel spacing (mm) from an affine transformation matrix.
    
    Args:
        affine: 4x4 affine transformation matrix from NIFTI header
        
    Returns:
        np.ndarray: Voxel spacings [sx, sy, sz] in millimeters
        
    Notes:
        Calculates spacing as the L2 norm of each directional column vector
        in the upper-left 3x3 rotation/scaling matrix.
    Example:
        R = 
            [[ 0.68,  0.00,  0.18],
             [ 0.00,  0.70,  0.00],
             [-0.18,  0.00,  1.18]]
        output: [0.705, 0.700, 1.193]
    """
    R = affine[:3, :3]
    sx = np.linalg.norm(R[:, 0]); sy = np.linalg.norm(R[:, 1]); sz = np.linalg.norm(R[:, 2])
    return np.array([sx, sy, sz], dtype=float)


def directions_from_affine(affine: np.ndarray) -> np.ndarray:
    """Extract normalized direction vectors from an affine transformation matrix.
    
    Args:
        affine: 4x4 affine transformation matrix
        
    Returns:
        np.ndarray: 3x3 matrix of normalized direction vectors
        
    Notes:
        Divides each column by its spacing to get unit vectors.
        Handles zero spacing edge case by replacing with 1.0.
    Example:
        R = 
            [[ 0.68,  0.00,  0.18],
             [ 0.00,  0.70,  0.00],
             [-0.18,  0.00,  1.18]].
        spacing_from_affine: [0.705, 0.700, 1.193], now divided by spacing gives
        final output: [[ 0.964,  0.000,  0.151],
                       [ 0.000,  1.000,  0.000],
                       [-0.255,  0.000,  0.989]]

    """
    R = affine[:3, :3]
    spac = spacing_from_affine(affine)
    spac = np.where(spac == 0, 1.0, spac)  # Avoid division by zero
    return R / spac


def make_affine_with_new_spacing(original_affine: np.ndarray, new_spacing) -> np.ndarray:
    """Generate new affine matrix with updated voxel spacing.
    
    Args:
        original_affine: Original 4x4 affine matrix
        new_spacing: Desired voxel spacing [sx, sy, sz] in mm
        
    Returns:
        np.ndarray: New 4x4 affine matrix with updated spacing
        
    Notes:
        Preserves orientation and origin from original affine.
        Only updates the scaling component of the transformation.
    """
    new_spacing = np.asarray(new_spacing, dtype=float)
    dirs = directions_from_affine(original_affine)
    R_new = dirs @ np.diag(new_spacing)
    out = np.eye(4, dtype=float)
    out[:3, :3] = R_new
    out[:3, 3]  = original_affine[:3, 3]  # Preserve original origin
    return out


# ---------- NIFTI File I/O Operations ----------

def load_ct_scan(path):
    """Load a NIFTI format medical image file.
    
    Args:
        path: Path to .nii or .nii.gz file
        
    Returns:
        tuple: (
            volume: np.ndarray of voxel data as float32,
            affine: 4x4 transformation matrix,
            header: NIfTI header object,
            img: Original NIfTI image object
        ) or (None, None, None, None) on error
        
    Notes:
        Loads both image data and geometric metadata.
        Converts voxel data to float32 for processing.
    """
    try:
        img = nib.load(path)
        return img.get_fdata(dtype=np.float32), img.affine, img.header, img
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None, None, None


def save_nifti(volume, affine, output_path, like_img=None, dtype=np.float32):
    """Save a volume as a NIFTI format file.
    
    Args:
        volume: 3D numpy array of voxel data
        affine: 4x4 transformation matrix
        output_path: Path for output .nii.gz file
        like_img: Optional template NIfTI image to copy header info from
        dtype: Data type for saving (default: float32)
        
    Notes:
        Creates output directory if needed.
        If template image provided, copies coordinate system codes
        and units to maintain geometric consistency.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = nib.Nifti1Image(volume.astype(dtype, copy=False), affine)
    if like_img is not None:
        try:
            xyzt = like_img.header.get_xyzt_units()
            img.header.set_xyzt_units(*xyzt)
        except Exception:
            pass
        # Set both qform and sform to maintain geometric consistency
        img.set_qform(affine, code=1)
        img.set_sform(affine, code=1)
    nib.save(img, output_path)
    print(f"✅ Saved: {output_path}")


# ---------- CT Image Preprocessing (HU Values) ----------

def preprocess_image(volume, affine, new_spacing=(1.0,1.0,1.0), hu_min=-100, hu_max=1000, interp_order=1):
    """Preprocess a CT volume: resample, clip HU values, and normalize.
    
    Args:
        volume: Input CT volume (3D numpy array in HU units)
        affine: 4x4 affine matrix defining the volume's geometry
        new_spacing: Target voxel spacing in mm (default: 1mm isotropic)
        hu_min: Lower HU clipping value (default: -100)
        hu_max: Upper HU clipping value (default: 1000)
        interp_order: Interpolation order (1=linear, default)
        
    Returns:
        tuple: (
            processed_volume: Normalized volume [0,1],
            new_affine: Updated geometry matrix,
            zoom_factors: Applied resampling factors
        )
        
    Processing steps:
    1. Clip HU values to [hu_min, hu_max] range
    2. Resample to target spacing using specified interpolation
    3. Normalize to [0,1] range
    4. Handle edge cases (NaN, inf) and convert to float32
    
    Notes:
        - Linear interpolation (order=1) recommended for CT data
        - HU window chosen for coronary artery visibility
        - Maintains geometric consistency via affine transforms
    """
    # Clip HU values to specified window
    vol = np.clip(volume, hu_min, hu_max).astype(np.float32, copy=False)

    # Calculate resampling factors and output shape
    orig_spacing = spacing_from_affine(affine)
    new_spacing = np.asarray(new_spacing, dtype=float)
    zoom, out_shape = _compute_zoom_and_outshape(orig_spacing, new_spacing, vol.shape)

    # Resample volume using torch (GPU if available)
    mode = "trilinear" if interp_order >= 1 else "nearest"
    vol_res = _torch_resample_3d(vol, out_shape, mode=mode)

    # Update geometric transform
    new_aff = make_affine_with_new_spacing(affine, new_spacing)

    # Normalize to [0,1] range
    vol_res = (vol_res - hu_min) / float(hu_max - hu_min)

    # Handle edge cases and ensure float32
    vol_res = np.nan_to_num(vol_res, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    return vol_res, new_aff, zoom


# ---------- Binary Mask Preprocessing ----------

def preprocess_mask(mask, affine, new_spacing=(1.0,1.0,1.0)):
    """Preprocess a binary segmentation mask with topology preservation.
    
    Args:
        mask: Input binary mask volume (3D numpy array)
        affine: 4x4 affine matrix defining the mask's geometry
        new_spacing: Target voxel spacing in mm (default: 1mm isotropic)
        
    Returns:
        tuple: (
            resampled_mask: Binary mask at new resolution,
            new_affine: Updated geometry matrix,
            zoom_factors: Applied resampling factors
        )
        
    Processing steps:
    1. Resample using nearest neighbor interpolation (preserves binary nature)
    2. Threshold at 0.5 to ensure strict binary values
    3. Convert to uint8 for memory efficiency
    
    Notes:
        - No intensity normalization needed (already binary)
        - Nearest neighbor interpolation preserves mask topology
        - Final thresholding ensures mask remains strictly binary
        - Maintains geometric consistency with corresponding image
    """
    mask = mask.astype(np.float32, copy=False)

    # Calculate resampling factors and output shape
    orig_spacing = spacing_from_affine(affine)
    new_spacing = np.asarray(new_spacing, dtype=float)
    zoom, out_shape = _compute_zoom_and_outshape(orig_spacing, new_spacing, mask.shape)

    # Nearest neighbor resampling via torch
    mask_res = _torch_resample_3d(mask, out_shape, mode="nearest")

    # Update geometric transform
    new_aff = make_affine_with_new_spacing(affine, new_spacing)

    # Ensure strict binary values and convert to uint8
    mask_res = (mask_res > 0.5).astype(np.uint8)

    return mask_res, new_aff, zoom


# ---------- Validation and Quality Control ----------

def summarize_volume(name, vol):
    """Generate statistical summary of a volume.
    
    Args:
        name: Identifier for the volume
        vol: 3D numpy array to analyze
        
    Returns:
        dict: Statistical metrics including shape, min, max, mean, std
        For smaller volumes (<2M voxels), also includes unique values
    """
    return {
        "name": name,
        "shape": tuple(vol.shape),
        "min": float(np.min(vol)),
        "max": float(np.max(vol)),
        "mean": float(np.mean(vol)),
        "std": float(np.std(vol)),
        "unique": None if vol.size > 2_000_000 else np.unique(vol)  # avoid huge prints
    }


def expected_shape_from_zoom(shape, zoom):
    """Calculate expected output shape after resampling.
    
    Args:
        shape: Original volume shape
        zoom: Resampling factors
    
    Returns:
        tuple: Expected shape after resampling
    """
    return tuple(int(round(s * z)) for s, z in zip(shape, zoom))


def verify_geometry(raw_aff, proc_aff, target_spacing, atol=ATOL):
    """Verify geometric consistency of preprocessing.
    
    Args:
        raw_aff: Original affine matrix
        proc_aff: Processed affine matrix
        target_spacing: Desired voxel spacing
        atol: Absolute tolerance for comparisons
        
    Returns:
        dict: Verification results including:
            - spacing_ok: Matches target within tolerance
            - orientation_ok: Preserved original directions
            - origin_ok: Preserved original origin
    """
    out = {}
    new_sp = spacing_from_affine(proc_aff)
    out["spacing_ok"] = bool(np.allclose(new_sp, np.asarray(target_spacing), atol=atol))
    out["new_spacing"] = tuple(np.round(new_sp, 4))
    out["orientation_ok"] = bool(np.allclose(directions_from_affine(raw_aff),
                                             directions_from_affine(proc_aff), atol=atol))
    out["origin_ok"] = bool(np.allclose(raw_aff[:3,3], proc_aff[:3,3], atol=atol))
    return out


def verify_image_values(proc_vol, atol=ATOL):
    """Verify normalized image value constraints.
    
    Args:
        proc_vol: Processed volume to verify
        atol: Absolute tolerance for range checks
        
    Returns:
        dict: Verification results including:
            - no_nans: True if no NaN/inf values
            - range_ok: True if values in [0,1] ± tolerance
    """
    return {
        "no_nans": bool(np.isfinite(proc_vol).all()),
        "range_ok": (float(proc_vol.min()) >= -atol) and (float(proc_vol.max()) <= 1.0 + atol),
    }


def verify_mask_values(mask_vol):
    """Verify binary mask constraints.
    
    Args:
        mask_vol: Mask volume to verify
        
    Returns:
        dict: Verification results including:
            - mask_binary: True if only 0s and 1s present
            - unique_vals: List of unique values found
    """
    uniq = np.unique(mask_vol)
    return {
        "mask_binary": set(uniq.tolist()).issubset({0,1}),
        "unique_vals": uniq.tolist()[:10]
    }


def print_stats(stats):
    """Pretty print volume statistics.
    
    Args:
        stats: Dict from summarize_volume()
    """
    u = "" if stats["unique"] is None else f" uniques={stats['unique']}"
    print(f"  {stats['name']}: shape={stats['shape']} min={stats['min']:.3f} "
          f"max={stats['max']:.3f} mean={stats['mean']:.3f} std={stats['std']:.3f}{u}")


def log_to_csv(csv_path, row, header):
    """Log processing results to CSV file.
    
    Args:
        csv_path: Output CSV path
        row: Dict of values to log
        header: List of column names
        
    Notes:
        Creates new file with header if doesn't exist,
        otherwise appends to existing file
    """
    first = not Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if first: w.writeheader()
        w.writerow(row)


# =========================
# MAIN (first N in folder)
# =========================

if __name__ == "__main__":
    start_time = time.time()
    INPUT_DIR   = Path(r"data\imageCAS_data\801-1000")
    OUTPUT_DIR  = Path(r"data\processed\801-1000")
    LOG_DIR     = Path(r"logs"); LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_CSV     = LOG_DIR / "preprocess_log.csv"

    N = 399
    TARGET_SPACING = (1.0, 1.0, 1.0)
    HU_MIN, HU_MAX = -100, 1000
    INTERP_ORDER   = 1  # 1 → trilinear (images), masks always use nearest

    files = sorted(list(INPUT_DIR.glob("*.nii")) + list(INPUT_DIR.glob("*.nii.gz")))
    if not files:
        print(f"No NIfTI files in: {INPUT_DIR}"); raise SystemExit(1)

    to_process = files[:N]
    print(f"Found {len(files)} files. Will process first {len(to_process)}.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    header = [
        "filename","type","raw_shape","proc_shape","expected_shape",
        "raw_min","raw_max","raw_mean","raw_std",
        "proc_min","proc_max","proc_mean","proc_std",
        "orig_spacing","new_spacing","target_spacing",
        "spacing_ok","shape_ok","orientation_ok","origin_ok",
        "values_ok","extra"
    ]

    for i, in_path in enumerate(to_process, 1):
        is_mask = (".label." in in_path.name.lower()) or in_path.name.lower().endswith("_mask.nii.gz")
        typ = "mask" if is_mask else "image"
        out_path = OUTPUT_DIR / in_path.name
        print(f"\n[{i}/{len(to_process)}] {in_path.name} ({typ})")

        raw_vol, raw_aff, hdr, like_img = load_ct_scan(str(in_path))
        if raw_vol is None:
            print("  ❌ Load failed"); continue

        if VERBOSE:
            print(f"  Original spacing: {tuple(np.round(spacing_from_affine(raw_aff),4))}")
            print_stats(summarize_volume("RAW", raw_vol))

        if is_mask:
            proc_vol, proc_aff, zoom = preprocess_mask(raw_vol, raw_aff, new_spacing=TARGET_SPACING)
        else:
            proc_vol, proc_aff, zoom = preprocess_image(
                raw_vol, raw_aff,
                new_spacing=TARGET_SPACING,
                hu_min=HU_MIN, hu_max=HU_MAX,
                interp_order=INTERP_ORDER
            )

        if VERBOSE:
            print(f"  New spacing     : {tuple(np.round(spacing_from_affine(proc_aff),4))}")
            print_stats(summarize_volume("PROC", proc_vol))

        checks_geo = verify_geometry(raw_aff, proc_aff, TARGET_SPACING, atol=ATOL)
        exp_shape = expected_shape_from_zoom(raw_vol.shape, zoom)
        shape_ok = np.all(np.abs(np.array(proc_vol.shape) - np.array(exp_shape)) <= 1)

        if is_mask:
            ch_vals = verify_mask_values(proc_vol)
            values_ok = ch_vals["mask_binary"]
            extra = f"unique={ch_vals['unique_vals']}"
            dtype = np.uint8
        else:
            ch_vals = verify_image_values(proc_vol, atol=ATOL)
            values_ok = ch_vals["no_nans"] and ch_vals["range_ok"]
            extra = ""
            dtype = np.float32

        print(f"  Checks → spacing:{checks_geo['spacing_ok']} | shape:{bool(shape_ok)} | "
              f"orient:{checks_geo['orientation_ok']} | origin:{checks_geo['origin_ok']} | "
              f"values_ok:{values_ok} {extra}")

        save_nifti(proc_vol, proc_aff, str(out_path), like_img=like_img, dtype=dtype)

        raw_stats  = summarize_volume("RAW", raw_vol)
        proc_stats = summarize_volume("PROC", proc_vol)
        row = {
            "filename": in_path.name,
            "type": typ,
            "raw_shape": raw_stats["shape"], "proc_shape": proc_stats["shape"],
            "expected_shape": tuple(exp_shape),
            "raw_min": raw_stats["min"], "raw_max": raw_stats["max"],
            "raw_mean": raw_stats["mean"], "raw_std": raw_stats["std"],
            "proc_min": proc_stats["min"], "proc_max": proc_stats["max"],
            "proc_mean": proc_stats["mean"], "proc_std": proc_stats["std"],
            "orig_spacing": tuple(np.round(spacing_from_affine(raw_aff),4)),
            "new_spacing": checks_geo["new_spacing"],
            "target_spacing": TARGET_SPACING,
            "spacing_ok": checks_geo["spacing_ok"],
            "shape_ok": bool(shape_ok),
            "orientation_ok": checks_geo["orientation_ok"],
            "origin_ok": checks_geo["origin_ok"],
            "values_ok": values_ok,
            "extra": extra
        }
        log_to_csv(LOG_CSV, row, header)

    print(f"\n✅ Done. Log: {LOG_CSV.resolve()}")
    end_time = time.time()
    print(f"⏱️ Total preprocessing time: {(end_time - start_time)/60:.2f} minutes")

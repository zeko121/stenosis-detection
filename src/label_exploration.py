"""
Label Spatial Analysis for CCTA Dataset

This script analyzes the spatial distribution of coronary artery masks to optimize
training pipeline parameters. It helps answer:
- Where do masks typically appear in the volume?
- What's the typical size of the region of interest?
- What target shape would capture 95%+ of cases efficiently?

Usage:
    python src/label_exploration.py --input_dirs data/imageCAS_data/801-1000 --out_dir outputs/label_analysis

Output:
- Bounding box size distributions (voxels and mm)
- Centroid spatial distributions (3D scatter + 2D heatmaps)
- Recommended target shape based on percentile analysis
- Summary statistics in JSON and human-readable formats
"""

from pathlib import Path
import argparse
import numpy as np
import nibabel as nib
import matplotlib
# Ensure backend is set before importing pyplot (headless mode)
matplotlib.use('Agg')  # Headless mode
import matplotlib.pyplot as plt
# seaborn is optional; use if available
try:
    import seaborn as sns
except Exception:
    sns = None
import json
from typing import List, Tuple, Dict, Any
import warnings

# ------------------ Helpers ------------------

def is_label_file(p: Path) -> bool:
    """Check if file is a label/mask based on naming convention."""
    name = p.name.lower()
    return (".label." in name) or name.endswith("_mask.nii") or name.endswith("_mask.nii.gz")


def base_name(p: Path) -> str:
    """Extract base filename without NIfTI extensions."""
    n = p.name
    if n.endswith('.nii.gz'):
        return n[:-7]
    if n.endswith('.nii'):
        return n[:-4]
    return p.stem


def spacing_from_affine(affine: np.ndarray) -> np.ndarray:
    """Extract voxel spacing (mm) from affine transformation matrix."""
    R = affine[:3, :3]
    sx = np.linalg.norm(R[:, 0])
    sy = np.linalg.norm(R[:, 1])
    sz = np.linalg.norm(R[:, 2])
    return np.array([sx, sy, sz], dtype=float)


# ------------------ Core Analysis ------------------

def extract_mask_properties(mask_path: Path) -> Dict[str, Any]:
    """Extract spatial properties from a single mask file.
    
    Returns dict with:
    - bbox_min_vox, bbox_max_vox: bounding box in voxel coordinates
    - bbox_size_vox, bbox_size_mm: bounding box dimensions
    - centroid_vox, centroid_mm: center of mass
    - volume_voxels: number of foreground voxels
    - volume_fraction: fraction of volume that is foreground
    - original_shape: original volume dimensions
    - spacing: voxel spacing in mm
    """
    try:
        img = nib.load(str(mask_path))
        arr = img.get_fdata(dtype=np.float32)
        affine = img.affine
        spacing = spacing_from_affine(affine)
    except Exception as e:
        print(f'  ✗ Failed to load {mask_path.name}: {e}')
        return None

    # Binary mask
    mask = (arr > 0.5)
    original_shape = tuple(arr.shape)
    
    # Find non-zero voxels
    nz = np.argwhere(mask)
    
    if nz.size == 0:
        # Empty mask
        return {
            'filename': mask_path.name,
            'original_shape': original_shape,
            'spacing': tuple(spacing),
            'is_empty': True,
            'bbox_min_vox': (0, 0, 0),
            'bbox_max_vox': (0, 0, 0),
            'bbox_size_vox': (0, 0, 0),
            'bbox_size_mm': (0.0, 0.0, 0.0),
            'centroid_vox': (np.nan, np.nan, np.nan),
            'centroid_mm': (np.nan, np.nan, np.nan),
            'volume_voxels': 0,
            'volume_fraction': 0.0
        }
    
    # Bounding box
    mins = nz.min(axis=0)
    maxs = nz.max(axis=0)
    sizes_vox = maxs - mins + 1
    sizes_mm = sizes_vox * spacing
    
    # Centroid (center of mass)
    centroid_vox = nz.mean(axis=0)
    centroid_mm = centroid_vox * spacing
    
    # Volume metrics
    volume_voxels = int(mask.sum())
    volume_fraction = float(volume_voxels / mask.size)
    
    return {
        'filename': mask_path.name,
        'original_shape': tuple(original_shape),
        'spacing': tuple(spacing),
        'is_empty': False,
        'bbox_min_vox': tuple(mins.tolist()),
        'bbox_max_vox': tuple(maxs.tolist()),
        'bbox_size_vox': tuple(sizes_vox.tolist()),
        'bbox_size_mm': tuple(sizes_mm.tolist()),
        'centroid_vox': tuple(centroid_vox.tolist()),
        'centroid_mm': tuple(centroid_mm.tolist()),
        'volume_voxels': volume_voxels,
        'volume_fraction': volume_fraction
    }


def compute_statistics(data: np.ndarray, axis: int = 0) -> Dict[str, Any]:
    """Compute summary statistics along specified axis."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return {
            'min': np.nanmin(data, axis=axis).tolist(),
            'max': np.nanmax(data, axis=axis).tolist(),
            'mean': np.nanmean(data, axis=axis).tolist(),
            'median': np.nanmedian(data, axis=axis).tolist(),
            'std': np.nanstd(data, axis=axis).tolist(),
            'p25': np.nanpercentile(data, 25, axis=axis).tolist(),
            'p75': np.nanpercentile(data, 75, axis=axis).tolist(),
            'p90': np.nanpercentile(data, 90, axis=axis).tolist(),
            'p95': np.nanpercentile(data, 95, axis=axis).tolist(),
            'p99': np.nanpercentile(data, 99, axis=axis).tolist()
        }


def propose_target_shape(bbox_sizes: np.ndarray, 
                         percentile: float = 95.0,
                         padding_vox: int = 8,
                         round_to: int = 16) -> Tuple[int, int, int]:
    """Propose optimal target shape based on bbox distribution.
    
    Args:
        bbox_sizes: Nx3 array of bounding box sizes
        percentile: Percentile to capture (e.g., 95 = captures 95% of cases)
        padding_vox: Safety padding to add per axis
        round_to: Round up to nearest multiple (for GPU efficiency)
        
    Returns:
        Proposed target shape as (x, y, z)
    """
    target = np.nanpercentile(bbox_sizes, percentile, axis=0)
    target = target + padding_vox
    
    # Round up to nearest multiple of round_to for GPU efficiency
    target = np.ceil(target / round_to).astype(int) * round_to
    
    return tuple(target.tolist())


# ------------------ Visualization ------------------

def create_visualizations(results: List[Dict], out_dir: Path, target_shape_vox: Tuple):
    """Generate comprehensive visualization suite."""
    
    # Extract arrays for plotting
    bbox_vox = np.array([r['bbox_size_vox'] for r in results if not r['is_empty']])
    bbox_mm = np.array([r['bbox_size_mm'] for r in results if not r['is_empty']])
    centroids_vox = np.array([r['centroid_vox'] for r in results if not r['is_empty']])
    centroids_mm = np.array([r['centroid_mm'] for r in results if not r['is_empty']])
    vol_fracs = np.array([r['volume_fraction'] for r in results if not r['is_empty']])
    
    axes_labels = ['X', 'Y', 'Z']
    
    # 1. Bounding box size distributions (voxels)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(3):
        axs[i].hist(bbox_vox[:, i], bins=50, alpha=0.7, edgecolor='black')
        axs[i].axvline(target_shape_vox[i], color='red', linestyle='--', 
                      linewidth=2, label=f'Target: {target_shape_vox[i]}')
        axs[i].set_xlabel(f'{axes_labels[i]} (voxels)')
        axs[i].set_ylabel('Count')
        axs[i].set_title(f'BBox Size Distribution - {axes_labels[i]} axis')
        axs[i].legend()
        axs[i].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'bbox_size_histograms_vox.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Bounding box size distributions (mm)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(3):
        axs[i].hist(bbox_mm[:, i], bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        axs[i].set_xlabel(f'{axes_labels[i]} (mm)')
        axs[i].set_ylabel('Count')
        axs[i].set_title(f'BBox Physical Size - {axes_labels[i]} axis')
        axs[i].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'bbox_size_histograms_mm.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 3. Centroid scatter plots (2D projections)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # XY plane
    axs[0].scatter(centroids_vox[:, 0], centroids_vox[:, 1], 
                   s=10, alpha=0.5, c=centroids_vox[:, 2], cmap='viridis')
    axs[0].set_xlabel('X (voxels)')
    axs[0].set_ylabel('Y (voxels)')
    axs[0].set_title('Centroid Distribution - XY plane (colored by Z)')
    axs[0].grid(True, alpha=0.3)
    
    # XZ plane
    axs[1].scatter(centroids_vox[:, 0], centroids_vox[:, 2], 
                   s=10, alpha=0.5, c=centroids_vox[:, 1], cmap='plasma')
    axs[1].set_xlabel('X (voxels)')
    axs[1].set_ylabel('Z (voxels)')
    axs[1].set_title('Centroid Distribution - XZ plane (colored by Y)')
    axs[1].grid(True, alpha=0.3)
    
    # YZ plane
    axs[2].scatter(centroids_vox[:, 1], centroids_vox[:, 2], 
                   s=10, alpha=0.5, c=centroids_vox[:, 0], cmap='coolwarm')
    axs[2].set_xlabel('Y (voxels)')
    axs[2].set_ylabel('Z (voxels)')
    axs[2].set_title('Centroid Distribution - YZ plane (colored by X)')
    axs[2].grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_dir / 'centroid_scatter_3views.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 4. Centroid density heatmaps
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # XY heatmap
    H_xy, xedges, yedges = np.histogram2d(centroids_vox[:, 0], centroids_vox[:, 1], bins=64)
    im0 = axs[0].imshow(H_xy.T, origin='lower', cmap='hot', interpolation='bilinear', aspect='auto')
    axs[0].set_title('Centroid Density - XY plane')
    axs[0].set_xlabel('X bin')
    axs[0].set_ylabel('Y bin')
    plt.colorbar(im0, ax=axs[0], label='count')
    
    # XZ heatmap
    H_xz, xedges, zedges = np.histogram2d(centroids_vox[:, 0], centroids_vox[:, 2], bins=64)
    im1 = axs[1].imshow(H_xz.T, origin='lower', cmap='hot', interpolation='bilinear', aspect='auto')
    axs[1].set_title('Centroid Density - XZ plane')
    axs[1].set_xlabel('X bin')
    axs[1].set_ylabel('Z bin')
    plt.colorbar(im1, ax=axs[1], label='count')
    
    # YZ heatmap
    H_yz, yedges, zedges = np.histogram2d(centroids_vox[:, 1], centroids_vox[:, 2], bins=64)
    im2 = axs[2].imshow(H_yz.T, origin='lower', cmap='hot', interpolation='bilinear', aspect='auto')
    axs[2].set_title('Centroid Density - YZ plane')
    axs[2].set_xlabel('Y bin')
    axs[2].set_ylabel('Z bin')
    plt.colorbar(im2, ax=axs[2], label='count')
    
    fig.tight_layout()
    fig.savefig(out_dir / 'centroid_heatmaps_3views.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 5. Volume fraction distribution
    plt.figure(figsize=(8, 5))
    plt.hist(vol_fracs * 100, bins=50, alpha=0.7, edgecolor='black', color='coral')
    plt.xlabel('Volume Fraction (%)')
    plt.ylabel('Count')
    plt.title('Mask Volume Fraction Distribution')
    plt.axvline(np.median(vol_fracs) * 100, color='red', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(vol_fracs)*100:.3f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / 'volume_fraction_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. Box plot summary
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        axs[i].boxplot([bbox_vox[:, i]], labels=[axes_labels[i]], vert=True)
        axs[i].axhline(target_shape_vox[i], color='red', linestyle='--', 
                      linewidth=2, label=f'Target: {target_shape_vox[i]}')
        axs[i].set_ylabel('Size (voxels)')
        axs[i].set_title(f'BBox Size Distribution - {axes_labels[i]}')
        axs[i].legend()
        axs[i].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'bbox_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f'  ✓ All visualizations saved to {out_dir}')


# ------------------ Main Analysis Pipeline ------------------

def analyze_labels(input_dirs: List[str], 
                   out_dir: Path,
                   max_files: int = None,
                   percentile: float = 95.0,
                   padding_vox: int = 8,
                   round_to: int = 16) -> None:
    """Main analysis routine.
    
    Args:
        input_dirs: Directories to scan for label files
        out_dir: Output directory for results
        max_files: Optional limit for faster dry-runs
        percentile: Percentile for target shape proposal (e.g., 95.0)
        padding_vox: Extra voxels to add to proposed shape
        round_to: Round target shape to nearest multiple (GPU efficiency)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all label files
    files = []
    for d in input_dirs:
        p = Path(d)
        if not p.exists():
            print(f'  ⚠ Directory not found: {p}')
            continue
        found = sorted([f for f in p.rglob('*.nii*') if is_label_file(f)])
        files.extend(found)
        print(f'  Found {len(found)} label files in {p}')
    
    if not files:
        print('✗ No label files found in the provided directories.')
        return
    
    if max_files:
        files = files[:max_files]
        print(f'  Limited to first {max_files} files for analysis.')
    
    print(f'\n▶ Analyzing {len(files)} label files...\n')
    
    # Extract properties from all masks
    results = []
    for i, f in enumerate(files, 1):
        props = extract_mask_properties(f)
        if props:
            results.append(props)
        if i % 50 == 0:
            print(f'  Progress: {i}/{len(files)}')
    
    print(f'\n✓ Successfully analyzed {len(results)}/{len(files)} files\n')
    
    # Filter out empty masks for statistics
    non_empty = [r for r in results if not r['is_empty']]
    n_empty = len(results) - len(non_empty)
    
    if not non_empty:
        print('✗ All masks are empty. Cannot perform analysis.')
        return
    
    # Convert to arrays
    bbox_vox_arr = np.array([r['bbox_size_vox'] for r in non_empty])
    bbox_mm_arr = np.array([r['bbox_size_mm'] for r in non_empty])
    centroids_vox_arr = np.array([r['centroid_vox'] for r in non_empty])
    centroids_mm_arr = np.array([r['centroid_mm'] for r in non_empty])
    vol_fracs = np.array([r['volume_fraction'] for r in non_empty])
    
    # Compute statistics
    bbox_stats_vox = compute_statistics(bbox_vox_arr, axis=0)
    bbox_stats_mm = compute_statistics(bbox_mm_arr, axis=0)
    centroid_stats_vox = compute_statistics(centroids_vox_arr, axis=0)
    centroid_stats_mm = compute_statistics(centroids_mm_arr, axis=0)
    vol_frac_stats = compute_statistics(vol_fracs)
    
    # Propose target shape
    target_shape_vox = propose_target_shape(bbox_vox_arr, percentile, padding_vox, round_to)
    
    # Calculate coverage
    covered = np.all(bbox_vox_arr <= np.array(target_shape_vox), axis=1)
    coverage_pct = (covered.sum() / len(non_empty)) * 100
    
    # Prepare summary
    summary = {
        'analysis_params': {
            'n_files_analyzed': len(results),
            'n_empty_masks': n_empty,
            'n_non_empty_masks': len(non_empty),
            'percentile_used': percentile,
            'padding_voxels': padding_vox,
            'round_to_multiple': round_to
        },
        'proposed_target_shape': {
            'voxels': target_shape_vox,
            'coverage_percentage': coverage_pct,
            'description': f'Captures {coverage_pct:.1f}% of non-empty masks'
        },
        'bbox_size_statistics': {
            'voxels': bbox_stats_vox,
            'millimeters': bbox_stats_mm
        },
        'centroid_statistics': {
            'voxels': centroid_stats_vox,
            'millimeters': centroid_stats_mm
        },
        'volume_fraction_statistics': vol_frac_stats,
        'sample_original_shapes': [r['original_shape'] for r in results[:20]]
    }
    
    # Save JSON summary
    json_path = out_dir / 'summary_stats.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save human-readable summary
    txt_path = out_dir / 'summary_stats.txt'
    with open(txt_path, 'w') as f:
        f.write('='*80 + '\n')
        f.write('CCTA Label Spatial Analysis Summary\n')
        f.write('='*80 + '\n\n')
        
        f.write(f'Files analyzed: {len(results)}\n')
        f.write(f'Empty masks: {n_empty}\n')
        f.write(f'Non-empty masks: {len(non_empty)}\n\n')
        
        f.write('-'*80 + '\n')
        f.write('RECOMMENDED TARGET SHAPE\n')
        f.write('-'*80 + '\n')
        f.write(f'Target shape (voxels): {target_shape_vox}\n')
        f.write(f'Coverage: {coverage_pct:.2f}% of cases\n')
        f.write(f'Based on: {percentile}th percentile + {padding_vox} voxel padding\n')
        f.write(f'Rounded to: nearest {round_to} voxels for GPU efficiency\n\n')
        
        f.write('-'*80 + '\n')
        f.write('BOUNDING BOX SIZE STATISTICS (voxels)\n')
        f.write('-'*80 + '\n')
        f.write(f"  Min:    {bbox_stats_vox['min']}\n")
        f.write(f"  P25:    {bbox_stats_vox['p25']}\n")
        f.write(f"  Median: {bbox_stats_vox['median']}\n")
        f.write(f"  Mean:   {bbox_stats_vox['mean']}\n")
        f.write(f"  P75:    {bbox_stats_vox['p75']}\n")
        f.write(f"  P95:    {bbox_stats_vox['p95']}\n")
        f.write(f"  Max:    {bbox_stats_vox['max']}\n")
        f.write(f"  Std:    {bbox_stats_vox['std']}\n\n")
        
        f.write('-'*80 + '\n')
        f.write('BOUNDING BOX SIZE STATISTICS (mm)\n')
        f.write('-'*80 + '\n')
        f.write(f"  Min:    {bbox_stats_mm['min']}\n")
        f.write(f"  Median: {bbox_stats_mm['median']}\n")
        f.write(f"  Mean:   {bbox_stats_mm['mean']}\n")
        f.write(f"  Max:    {bbox_stats_mm['max']}\n\n")
        
        f.write('-'*80 + '\n')
        f.write('CENTROID STATISTICS (voxels)\n')
        f.write('-'*80 + '\n')
        f.write(f"  Mean:   {centroid_stats_vox['mean']}\n")
        f.write(f"  Median: {centroid_stats_vox['median']}\n")
        f.write(f"  Std:    {centroid_stats_vox['std']}\n\n")
        
        f.write('-'*80 + '\n')
        f.write('VOLUME FRACTION STATISTICS\n')
        f.write('-'*80 + '\n')
        med_frac = vol_frac_stats['median'] if isinstance(vol_frac_stats['median'], float) else vol_frac_stats['median'][0]
        mean_frac = vol_frac_stats['mean'] if isinstance(vol_frac_stats['mean'], float) else vol_frac_stats['mean'][0]
        f.write(f"  Median: {med_frac*100:.4f}%\n")
        f.write(f"  Mean:   {mean_frac*100:.4f}%\n\n")
        
        f.write('='*80 + '\n')
        f.write('RECOMMENDATIONS\n')
        f.write('='*80 + '\n')
        f.write(f'1. Use target_shape = {target_shape_vox} for preprocessing\n')
        f.write(f'2. This captures {coverage_pct:.1f}% of cases without excessive padding\n')
        f.write(f'3. Outliers beyond this size will be center-cropped or padded\n')
        f.write(f'4. Consider using patch-based training if masks are very sparse\n')
        f.write(f'   (median volume fraction: {med_frac*100:.4f}%)\n')
    
    print(f'✓ Summary saved to:')
    print(f'  - {json_path}')
    print(f'  - {txt_path}')
    
    # Generate visualizations
    print(f'\n▶ Generating visualizations...')
    create_visualizations(non_empty, out_dir, target_shape_vox)
    
    print(f'\n{"="*80}')
    print(f'✓ Analysis complete!')
    print(f'{"="*80}')
    print(f'\nRecommended target_shape: {target_shape_vox}')
    print(f'Coverage: {coverage_pct:.1f}% of cases')
    print(f'\nAll outputs saved to: {out_dir}')


# ------------------ CLI ------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze spatial distribution of CCTA label masks and propose optimal target shape',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input_dirs', nargs='*', default=['data/imageCAS_data'],
                       help='Directories to scan for label NIfTI files (default: data/imageCAS_data)')
    parser.add_argument('--out_dir', default='outputs/label_analysis',
                       help='Output directory for results')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Limit number of files for quick test run')
    parser.add_argument('--percentile', type=float, default=95.0,
                       help='Percentile for target shape (e.g., 95 = capture 95%% of cases)')
    parser.add_argument('--padding', type=int, default=8,
                       help='Extra padding (voxels) added to proposed shape')
    parser.add_argument('--round_to', type=int, default=16,
                       help='Round target shape to nearest multiple (for GPU efficiency)')
    
    args = parser.parse_args()
    
    analyze_labels(
        input_dirs=args.input_dirs,
        out_dir=Path(args.out_dir),
        max_files=args.max_files,
        percentile=args.percentile,
        padding_vox=args.padding,
        round_to=args.round_to
    )
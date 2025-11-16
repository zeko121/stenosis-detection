"""
Visualization API Endpoints
Serves image slices, 3D data, and visualization metadata
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import nibabel as nib
import numpy as np
from pathlib import Path
from io import BytesIO
import zarr

from app.core.database import get_db
from app.core.config import settings
from app.models.database import Case
from app.models.schemas import SliceRequest, VolumeMetadata

router = APIRouter(prefix="/visualization", tags=["visualization"])


def apply_windowing(data: np.ndarray, window_min: int, window_max: int) -> np.ndarray:
    """Apply HU windowing to CT data"""
    windowed = np.clip(data, window_min, window_max)
    normalized = (windowed - window_min) / (window_max - window_min)
    return (normalized * 255).astype(np.uint8)


def get_slice_from_volume(
    volume: np.ndarray,
    plane: str,
    index: int
) -> np.ndarray:
    """
    Extract a 2D slice from 3D volume

    Args:
        volume: 3D numpy array
        plane: "axial", "coronal", or "sagittal"
        index: Slice index

    Returns:
        2D slice array
    """
    if plane == "axial":
        # Z-axis slice
        if index >= volume.shape[2]:
            raise ValueError(f"Index {index} out of range for {plane} (max: {volume.shape[2]-1})")
        return volume[:, :, index]

    elif plane == "coronal":
        # Y-axis slice
        if index >= volume.shape[1]:
            raise ValueError(f"Index {index} out of range for {plane} (max: {volume.shape[1]-1})")
        return volume[:, index, :]

    elif plane == "sagittal":
        # X-axis slice
        if index >= volume.shape[0]:
            raise ValueError(f"Index {index} out of range for {plane} (max: {volume.shape[0]-1})")
        return volume[index, :, :]

    else:
        raise ValueError(f"Invalid plane: {plane}")


@router.get("/{case_id}/metadata", response_model=VolumeMetadata)
async def get_volume_metadata(
    case_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get 3D volume metadata for visualization setup

    Args:
        case_id: Case identifier
        db: Database session

    Returns:
        VolumeMetadata with dimensions and orientation
    """
    result = await db.execute(select(Case).where(Case.case_id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Load NIFTI to get complete metadata
    try:
        nifti = nib.load(case.file_path)
        affine = nifti.affine
        origin = affine[:3, 3].tolist()

        num_slices = {
            "axial": case.dimensions[2],
            "coronal": case.dimensions[1],
            "sagittal": case.dimensions[0]
        }

        return VolumeMetadata(
            dimensions=case.dimensions,
            spacing=case.spacing,
            origin=origin,
            orientation=case.orientation or "RAS",
            num_slices=num_slices
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load metadata: {str(e)}")


@router.get("/{case_id}/slice")
async def get_slice(
    case_id: str,
    plane: str,
    index: int,
    window_min: int = -100,
    window_max: int = 1000,
    show_overlay: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a 2D slice from the volume

    Args:
        case_id: Case identifier
        plane: "axial", "coronal", or "sagittal"
        index: Slice index
        window_min: Minimum HU value for windowing
        window_max: Maximum HU value for windowing
        show_overlay: Whether to overlay segmentation
        db: Database session

    Returns:
        PNG image of the slice
    """
    if plane not in ["axial", "coronal", "sagittal"]:
        raise HTTPException(status_code=400, detail="Invalid plane")

    result = await db.execute(select(Case).where(Case.case_id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    try:
        # Check if preprocessed Zarr exists (faster loading)
        zarr_path = settings.PROCESSED_DIR / case_id / f"preprocessed_{Path(case.file_path).stem}.zarr"

        if zarr_path.exists():
            volume = zarr.load(str(zarr_path))
        else:
            # Load from original NIFTI
            nifti = nib.load(case.file_path)
            volume = nifti.get_fdata()

        # Extract slice
        slice_2d = get_slice_from_volume(volume, plane, index)

        # Apply windowing
        slice_windowed = apply_windowing(slice_2d, window_min, window_max)

        # Overlay segmentation if requested
        if show_overlay:
            seg_path = settings.PROCESSED_DIR / case_id / "segmentation.nii.gz"
            if seg_path.exists():
                seg_nifti = nib.load(str(seg_path))
                seg_volume = seg_nifti.get_fdata()
                seg_slice = get_slice_from_volume(seg_volume, plane, index)

                # Create RGB image with overlay
                rgb_image = np.stack([slice_windowed] * 3, axis=-1)

                # Color-code vessels
                colors = {
                    1: [255, 0, 0],    # LMCA - Red
                    2: [0, 255, 0],    # LAD - Green
                    3: [0, 0, 255],    # LCx - Blue
                    4: [255, 255, 0]   # RCA - Yellow
                }

                for vessel_id, color in colors.items():
                    mask = seg_slice == vessel_id
                    for c in range(3):
                        rgb_image[mask, c] = int(0.7 * rgb_image[mask, c] + 0.3 * color[c])

                slice_windowed = rgb_image.astype(np.uint8)

        # Convert to PNG
        from PIL import Image
        img = Image.fromarray(slice_windowed)

        # Save to bytes
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return StreamingResponse(img_bytes, media_type="image/png")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate slice: {str(e)}")


@router.get("/{case_id}/volume_data")
async def get_volume_data(
    case_id: str,
    downsample: int = 2,
    db: AsyncSession = Depends(get_db)
):
    """
    Get downsampled 3D volume data for volume rendering

    Args:
        case_id: Case identifier
        downsample: Downsampling factor (e.g., 2 = half resolution)
        db: Database session

    Returns:
        Binary volume data in numpy format
    """
    result = await db.execute(select(Case).where(Case.case_id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    try:
        # Load preprocessed data if available
        zarr_path = settings.PROCESSED_DIR / case_id / f"preprocessed_{Path(case.file_path).stem}.zarr"

        if zarr_path.exists():
            volume = zarr.load(str(zarr_path))
        else:
            nifti = nib.load(case.file_path)
            volume = nifti.get_fdata()

        # Downsample if requested
        if downsample > 1:
            volume = volume[::downsample, ::downsample, ::downsample]

        # Normalize to uint8
        volume_normalized = ((volume - volume.min()) / (volume.max() - volume.min()) * 255).astype(np.uint8)

        # Convert to bytes
        volume_bytes = BytesIO()
        np.save(volume_bytes, volume_normalized)
        volume_bytes.seek(0)

        return StreamingResponse(
            volume_bytes,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=volume_{case_id}.npy",
                "X-Volume-Shape": str(volume_normalized.shape)
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load volume: {str(e)}")


@router.get("/{case_id}/segmentation_mask")
async def get_segmentation_mask(
    case_id: str,
    vessel: str = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Get segmentation mask for 3D visualization

    Args:
        case_id: Case identifier
        vessel: Specific vessel name (LMCA, LAD, LCx, RCA) or None for all
        db: Database session

    Returns:
        Binary segmentation mask
    """
    result = await db.execute(select(Case).where(Case.case_id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    seg_path = settings.PROCESSED_DIR / case_id / "segmentation.nii.gz"

    if not seg_path.exists():
        raise HTTPException(status_code=404, detail="Segmentation not available")

    try:
        seg_nifti = nib.load(str(seg_path))
        seg_volume = seg_nifti.get_fdata().astype(np.uint8)

        # Filter by vessel if specified
        if vessel:
            vessel_map = {"LMCA": 1, "LAD": 2, "LCx": 3, "RCA": 4}
            if vessel not in vessel_map:
                raise HTTPException(status_code=400, detail=f"Invalid vessel: {vessel}")

            seg_volume = (seg_volume == vessel_map[vessel]).astype(np.uint8)

        # Convert to bytes
        mask_bytes = BytesIO()
        np.save(mask_bytes, seg_volume)
        mask_bytes.seek(0)

        return StreamingResponse(
            mask_bytes,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=mask_{case_id}.npy",
                "X-Volume-Shape": str(seg_volume.shape)
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load segmentation: {str(e)}")

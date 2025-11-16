"""
Case Management API Endpoints
Handles file upload, case listing, status tracking, and results retrieval
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
from pathlib import Path
import shutil
import nibabel as nib
from datetime import datetime
import uuid

from app.core.database import get_db
from app.core.config import settings
from app.models.database import Case, CaseStatus
from app.models.schemas import (
    CaseUploadResponse,
    CaseListItem,
    CaseDetail,
    ProcessingConfig,
    ProcessingStatus,
    CaseResults,
    ErrorResponse
)
from app.services.tasks import process_case_task

router = APIRouter(prefix="/cases", tags=["cases"])


def generate_case_id() -> str:
    """Generate unique case ID"""
    timestamp = datetime.now().strftime("%Y%m%d")
    unique_id = str(uuid.uuid4())[:8].upper()
    return f"CAS-{timestamp}-{unique_id}"


async def validate_nifti_file(file_path: str) -> dict:
    """
    Validate NIFTI file and extract metadata

    Args:
        file_path: Path to NIFTI file

    Returns:
        Dictionary with file metadata

    Raises:
        ValueError: If file is invalid
    """
    try:
        nifti = nib.load(file_path)
        header = nifti.header
        affine = nifti.affine

        # Extract metadata
        dimensions = list(nifti.shape)
        spacing = list(header.get_zooms())
        orientation = nib.aff2axcodes(affine)

        # Basic validation
        if len(dimensions) != 3:
            raise ValueError(f"Expected 3D volume, got shape: {dimensions}")

        if any(d < 10 for d in dimensions):
            raise ValueError(f"Volume dimensions too small: {dimensions}")

        return {
            "dimensions": dimensions,
            "spacing": spacing,
            "orientation": "".join(orientation),
            "affine_matrix": affine.tolist(),
            "datatype": str(header.get_data_dtype())
        }

    except Exception as e:
        raise ValueError(f"Invalid NIFTI file: {str(e)}")


@router.post("/upload", response_model=CaseUploadResponse)
async def upload_case(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a new CCTA scan for processing

    Args:
        file: NIFTI file (.nii or .nii.gz)
        db: Database session

    Returns:
        CaseUploadResponse with case information
    """
    # Validate file extension
    file_ext = "".join(Path(file.filename).suffixes)
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )

    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset
    file_size_mb = file_size / (1024 * 1024)

    if file_size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE_MB}MB"
        )

    # Generate case ID
    case_id = generate_case_id()

    # Save file
    upload_dir = settings.UPLOAD_DIR / case_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Validate NIFTI and extract metadata
        metadata = await validate_nifti_file(str(file_path))

        # Create database record
        new_case = Case(
            case_id=case_id,
            original_filename=file.filename,
            file_path=str(file_path),
            file_size_mb=file_size_mb,
            dimensions=metadata["dimensions"],
            spacing=metadata["spacing"],
            orientation=metadata["orientation"],
            affine_matrix=metadata["affine_matrix"],
            status=CaseStatus.UPLOADED,
            progress=0.0
        )

        db.add(new_case)
        await db.commit()
        await db.refresh(new_case)

        return CaseUploadResponse(
            case_id=case_id,
            status=CaseStatus.UPLOADED,
            message="File uploaded successfully",
            upload_info={
                "filename": file.filename,
                "size_mb": round(file_size_mb, 2),
                "dimensions": metadata["dimensions"],
                "spacing": metadata["spacing"]
            }
        )

    except ValueError as e:
        # Delete uploaded file if validation fails
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("", response_model=List[CaseListItem])
async def list_cases(
    skip: int = 0,
    limit: int = 100,
    status: Optional[CaseStatus] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    List all cases with optional filtering

    Args:
        skip: Number of records to skip
        limit: Maximum records to return
        status: Filter by status
        db: Database session

    Returns:
        List of CaseListItem objects
    """
    query = select(Case).order_by(Case.created_at.desc())

    if status:
        query = query.where(Case.status == status)

    query = query.offset(skip).limit(limit)

    result = await db.execute(query)
    cases = result.scalars().all()

    return [
        CaseListItem(
            case_id=case.case_id,
            original_filename=case.original_filename,
            upload_date=case.created_at,
            status=case.status,
            progress=case.progress,
            dimensions=case.dimensions,
            spacing=case.spacing
        )
        for case in cases
    ]


@router.get("/{case_id}", response_model=CaseDetail)
async def get_case(case_id: str, db: AsyncSession = Depends(get_db)):
    """
    Get detailed information for a specific case

    Args:
        case_id: Case identifier
        db: Database session

    Returns:
        CaseDetail object
    """
    result = await db.execute(select(Case).where(Case.case_id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    return CaseDetail.model_validate(case)


@router.post("/{case_id}/process", response_model=ProcessingStatus)
async def process_case(
    case_id: str,
    config: ProcessingConfig = ProcessingConfig(),
    db: AsyncSession = Depends(get_db)
):
    """
    Start processing a case

    Args:
        case_id: Case identifier
        config: Processing configuration
        db: Database session

    Returns:
        ProcessingStatus with job information
    """
    # Get case from database
    result = await db.execute(select(Case).where(Case.case_id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    if case.status not in [CaseStatus.UPLOADED, CaseStatus.ERROR]:
        raise HTTPException(
            status_code=400,
            detail=f"Case cannot be processed in status: {case.status}"
        )

    # Create output directory
    output_dir = settings.PROCESSED_DIR / case_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Start Celery task
    task = process_case_task.delay(
        case_id=case_id,
        nifti_path=case.file_path,
        output_dir=str(output_dir),
        config=config.model_dump()
    )

    # Update case status
    case.status = CaseStatus.PREPROCESSING
    case.job_id = task.id
    case.processing_started_at = datetime.now()
    case.current_stage = "preprocessing"
    case.progress = 0.0

    await db.commit()

    return ProcessingStatus(
        case_id=case_id,
        status=CaseStatus.PREPROCESSING,
        progress=0.0,
        current_stage="preprocessing",
        estimated_time_remaining_seconds=180
    )


@router.get("/{case_id}/status", response_model=ProcessingStatus)
async def get_case_status(case_id: str, db: AsyncSession = Depends(get_db)):
    """
    Get current processing status for a case

    Args:
        case_id: Case identifier
        db: Database session

    Returns:
        ProcessingStatus object
    """
    result = await db.execute(select(Case).where(Case.case_id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # If processing, check Celery task status
    if case.job_id and case.status not in [CaseStatus.COMPLETE, CaseStatus.ERROR]:
        from app.services.tasks import celery_app
        task_result = celery_app.AsyncResult(case.job_id)

        if task_result.state == "SUCCESS":
            case.status = CaseStatus.COMPLETE
            case.progress = 1.0
            case.processing_completed_at = datetime.now()

            if case.processing_started_at:
                elapsed = (case.processing_completed_at - case.processing_started_at).total_seconds()
                case.processing_time_seconds = elapsed

            await db.commit()

        elif task_result.state == "FAILURE":
            case.status = CaseStatus.ERROR
            case.error_message = str(task_result.info)
            await db.commit()

        elif task_result.state in ["PREPROCESSING", "SEGMENTING", "CLASSIFYING"]:
            meta = task_result.info or {}
            case.current_stage = meta.get("stage", case.current_stage)
            case.progress = meta.get("progress", case.progress)
            await db.commit()

    # Estimate remaining time
    estimated_time = None
    if case.status not in [CaseStatus.COMPLETE, CaseStatus.ERROR] and case.progress > 0:
        if case.processing_started_at:
            elapsed = (datetime.now() - case.processing_started_at).total_seconds()
            total_estimated = elapsed / case.progress
            estimated_time = int(total_estimated - elapsed)

    return ProcessingStatus(
        case_id=case_id,
        status=case.status,
        progress=case.progress,
        current_stage=case.current_stage or "queued",
        estimated_time_remaining_seconds=estimated_time
    )


@router.delete("/{case_id}")
async def delete_case(case_id: str, db: AsyncSession = Depends(get_db)):
    """
    Delete a case and all associated files

    Args:
        case_id: Case identifier
        db: Database session

    Returns:
        Success message
    """
    result = await db.execute(select(Case).where(Case.case_id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Cancel running job if exists
    if case.job_id and case.status not in [CaseStatus.COMPLETE, CaseStatus.ERROR]:
        from app.services.tasks import celery_app
        celery_app.control.revoke(case.job_id, terminate=True)

    # Delete files
    try:
        upload_dir = settings.UPLOAD_DIR / case_id
        if upload_dir.exists():
            shutil.rmtree(upload_dir)

        processed_dir = settings.PROCESSED_DIR / case_id
        if processed_dir.exists():
            shutil.rmtree(processed_dir)

    except Exception as e:
        # Log error but continue with database deletion
        print(f"Error deleting files for {case_id}: {e}")

    # Delete database record
    await db.delete(case)
    await db.commit()

    return {"message": f"Case {case_id} deleted successfully"}

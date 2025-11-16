"""
Results API Endpoints
Retrieve analysis results and generate reports
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from pathlib import Path

from app.core.database import get_db
from app.core.config import settings
from app.models.database import Case, SegmentationResult, StenosisResult, CaseStatus
from app.models.schemas import (
    CaseResults,
    SegmentationResults,
    StenosisResults,
    ReportRequest,
    ReportResponse,
    VesselSegmentation,
    StenosisAnalysis
)
from app.services.tasks import generate_report_task

router = APIRouter(prefix="/results", tags=["results"])


@router.get("/{case_id}", response_model=CaseResults)
async def get_case_results(
    case_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get complete analysis results for a case

    Args:
        case_id: Case identifier
        db: Database session

    Returns:
        CaseResults with segmentation and stenosis analysis
    """
    # Get case
    result = await db.execute(select(Case).where(Case.case_id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    if case.status != CaseStatus.COMPLETE:
        raise HTTPException(
            status_code=400,
            detail=f"Case not complete. Current status: {case.status}"
        )

    # Get segmentation results
    seg_result = await db.execute(
        select(SegmentationResult).where(SegmentationResult.case_id == case_id)
    )
    seg = seg_result.scalar_one_or_none()

    segmentation_results = None
    if seg:
        vessels = [
            VesselSegmentation(
                vessel_name="LMCA",
                detected=bool(seg.lmca_detected),
                volume_mm3=seg.vessel_volume_mm3,
                confidence=None
            ),
            VesselSegmentation(
                vessel_name="LAD",
                detected=bool(seg.lad_detected),
                volume_mm3=None,
                confidence=None
            ),
            VesselSegmentation(
                vessel_name="LCx",
                detected=bool(seg.lcx_detected),
                volume_mm3=None,
                confidence=None
            ),
            VesselSegmentation(
                vessel_name="RCA",
                detected=bool(seg.rca_detected),
                volume_mm3=None,
                confidence=None
            )
        ]

        segmentation_results = SegmentationResults(
            case_id=case_id,
            segmentation_path=seg.segmentation_path or "",
            zarr_path=seg.zarr_path,
            dice_score=seg.dice_score,
            iou_score=seg.iou_score,
            vessels=vessels,
            model_name=seg.model_name or "UNet3D",
            model_version=seg.model_version or "1.0"
        )

    # Get stenosis results
    stenosis_result = await db.execute(
        select(StenosisResult).where(StenosisResult.case_id == case_id)
    )
    stenosis_records = stenosis_result.scalars().all()

    stenosis_results = None
    if stenosis_records:
        analyses = [
            StenosisAnalysis(
                vessel_name=record.vessel_name,
                vessel_segment=record.vessel_segment,
                stenosis_percentage=record.stenosis_percentage,
                severity_class=record.severity_class,
                confidence_score=record.confidence_score,
                stenosis_location_mm=record.stenosis_location_mm,
                reference_radius_mm=record.reference_radius_mm,
                minimum_radius_mm=record.minimum_radius_mm,
                severity_probabilities=record.severity_probabilities or {}
            )
            for record in stenosis_records
        ]

        # Find most severe stenosis
        most_severe = max(analyses, key=lambda x: x.stenosis_percentage)

        stenosis_results = StenosisResults(
            case_id=case_id,
            analyses=analyses,
            overall_severity=most_severe.severity_class,
            most_severe_vessel=most_severe.vessel_name,
            most_severe_stenosis=most_severe.stenosis_percentage
        )

    return CaseResults(
        case_id=case_id,
        status=case.status,
        segmentation=segmentation_results,
        stenosis=stenosis_results,
        processing_time_seconds=case.processing_time_seconds
    )


@router.post("/{case_id}/report", response_model=ReportResponse)
async def generate_report(
    case_id: str,
    request: ReportRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate a clinical report for a case

    Args:
        case_id: Case identifier
        request: Report generation parameters
        db: Database session

    Returns:
        ReportResponse with download information
    """
    # Verify case exists and is complete
    result = await db.execute(select(Case).where(Case.case_id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    if case.status != CaseStatus.COMPLETE:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot generate report for incomplete case. Status: {case.status}"
        )

    # Get results
    results = await get_case_results(case_id, db)

    # Generate output path
    report_filename = f"{case_id}_report.{request.format}"
    report_path = settings.REPORTS_DIR / report_filename

    # Start report generation task
    task = generate_report_task.delay(
        case_id=case_id,
        results=results.model_dump(),
        output_path=str(report_path),
        template=request.template
    )

    # Wait for task to complete (for MVP, could be async in production)
    task.get(timeout=30)

    return ReportResponse(
        case_id=case_id,
        report_path=str(report_path),
        download_url=f"/api/v1/results/{case_id}/download/{report_filename}",
        format=request.format
    )


@router.get("/{case_id}/download/{filename}")
async def download_report(
    case_id: str,
    filename: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Download a generated report

    Args:
        case_id: Case identifier
        filename: Report filename
        db: Database session

    Returns:
        File download response
    """
    # Verify case exists
    result = await db.execute(select(Case).where(Case.case_id == case_id))
    case = result.scalar_one_or_none()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Get report file
    report_path = settings.REPORTS_DIR / filename

    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")

    # Verify filename matches case_id for security
    if not filename.startswith(case_id):
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(
        path=report_path,
        filename=filename,
        media_type="application/pdf" if filename.endswith(".pdf") else "application/json"
    )

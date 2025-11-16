"""
Pydantic Schemas
Request and response models for API validation
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# Enums
class CaseStatusEnum(str, Enum):
    UPLOADED = "uploaded"
    VALIDATING = "validating"
    PREPROCESSING = "preprocessing"
    SEGMENTING = "segmenting"
    CLASSIFYING = "classifying"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


class StenosisSeverityEnum(str, Enum):
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


# Case Schemas
class CaseUploadResponse(BaseModel):
    """Response after successful file upload"""
    case_id: str
    status: CaseStatusEnum
    message: str
    upload_info: Dict[str, Any]


class CaseListItem(BaseModel):
    """Brief case information for list view"""
    case_id: str
    original_filename: str
    upload_date: datetime
    status: CaseStatusEnum
    progress: float
    dimensions: Optional[List[int]] = None
    spacing: Optional[List[float]] = None

    class Config:
        from_attributes = True


class CaseDetail(BaseModel):
    """Detailed case information"""
    case_id: str
    original_filename: str
    file_size_mb: Optional[float]
    status: CaseStatusEnum
    progress: float
    current_stage: Optional[str]
    error_message: Optional[str]

    # Metadata
    dimensions: Optional[List[int]]
    spacing: Optional[List[float]]
    orientation: Optional[str]
    scan_date: Optional[datetime]

    # Processing info
    job_id: Optional[str]
    processing_started_at: Optional[datetime]
    processing_completed_at: Optional[datetime]
    processing_time_seconds: Optional[float]

    # Timestamps
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Processing Schemas
class ProcessingConfig(BaseModel):
    """Configuration for processing a case"""
    use_gpu: bool = True
    batch_size: int = 1
    hu_window_min: int = -100
    hu_window_max: int = 1000
    target_spacing: float = 0.8
    enable_classification: bool = True


class ProcessingStatus(BaseModel):
    """Current processing status"""
    case_id: str
    status: CaseStatusEnum
    progress: float
    current_stage: str
    estimated_time_remaining_seconds: Optional[int]


# Results Schemas
class VesselSegmentation(BaseModel):
    """Segmentation result for a vessel"""
    vessel_name: str
    detected: bool
    volume_mm3: Optional[float]
    confidence: Optional[float]


class StenosisAnalysis(BaseModel):
    """Stenosis analysis for a vessel"""
    vessel_name: str
    vessel_segment: Optional[str]
    stenosis_percentage: float
    severity_class: StenosisSeverityEnum
    confidence_score: float
    stenosis_location_mm: Optional[float]
    reference_radius_mm: Optional[float]
    minimum_radius_mm: Optional[float]
    severity_probabilities: Dict[str, float]


class SegmentationResults(BaseModel):
    """Complete segmentation results"""
    case_id: str
    segmentation_path: str
    zarr_path: Optional[str]
    dice_score: Optional[float]
    iou_score: Optional[float]
    vessels: List[VesselSegmentation]
    model_name: str
    model_version: str


class StenosisResults(BaseModel):
    """Complete stenosis analysis results"""
    case_id: str
    analyses: List[StenosisAnalysis]
    overall_severity: StenosisSeverityEnum
    most_severe_vessel: str
    most_severe_stenosis: float


class CaseResults(BaseModel):
    """Complete results for a case"""
    case_id: str
    status: CaseStatusEnum
    segmentation: Optional[SegmentationResults]
    stenosis: Optional[StenosisResults]
    processing_time_seconds: Optional[float]


# Visualization Schemas
class SliceRequest(BaseModel):
    """Request for a specific slice"""
    plane: str = Field(..., pattern="^(axial|coronal|sagittal)$")
    index: int = Field(..., ge=0)
    window_min: Optional[int] = -100
    window_max: Optional[int] = 1000
    show_overlay: bool = False


class VolumeMetadata(BaseModel):
    """3D volume metadata for visualization"""
    dimensions: List[int]
    spacing: List[float]
    origin: List[float]
    orientation: str
    num_slices: Dict[str, int]  # {"axial": 200, "coronal": 512, "sagittal": 512}


# Report Schemas
class ReportRequest(BaseModel):
    """Request for report generation"""
    format: str = Field("pdf", pattern="^(pdf|json)$")
    template: str = Field("comprehensive", pattern="^(brief|comprehensive)$")
    include_3d_renderings: bool = True
    include_mpr_slices: bool = True
    physician_notes: Optional[str] = None


class ReportResponse(BaseModel):
    """Response after report generation"""
    case_id: str
    report_path: str
    download_url: str
    format: str


# Model Management Schemas
class ModelInfo(BaseModel):
    """Information about a trained model"""
    model_name: str
    model_type: str  # "segmentation" or "classification"
    version: str
    architecture: str
    training_dataset: str
    performance_metrics: Dict[str, float]
    trained_date: datetime
    file_size_mb: float
    is_active: bool


class ModelUploadResponse(BaseModel):
    """Response after model upload"""
    model_name: str
    message: str
    model_info: ModelInfo


# Error Schemas
class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: str
    case_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

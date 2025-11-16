"""
Database Models
SQLAlchemy models for case management and results storage
"""
from sqlalchemy import Column, Integer, String, DateTime, Float, JSON, Enum as SQLEnum, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import enum


Base = declarative_base()


class CaseStatus(str, enum.Enum):
    """Case processing status"""
    UPLOADED = "uploaded"
    VALIDATING = "validating"
    PREPROCESSING = "preprocessing"
    SEGMENTING = "segmenting"
    CLASSIFYING = "classifying"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


class StenosisSeverity(str, enum.Enum):
    """Stenosis severity levels"""
    NORMAL = "normal"          # 0-25%
    MILD = "mild"              # 25-50%
    MODERATE = "moderate"      # 50-70%
    SEVERE = "severe"          # >70%


class Case(Base):
    """Main case table for uploaded scans"""
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String(50), unique=True, index=True, nullable=False)  # e.g., "CAS-001"

    # File Information
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size_mb = Column(Float)

    # NIFTI Metadata
    dimensions = Column(JSON)  # [x, y, z]
    spacing = Column(JSON)     # [sx, sy, sz]
    orientation = Column(String(50))
    affine_matrix = Column(JSON)

    # Patient Information (anonymized)
    scan_date = Column(DateTime, nullable=True)
    patient_id = Column(String(100), nullable=True)  # Anonymized ID

    # Processing Status
    status = Column(SQLEnum(CaseStatus), default=CaseStatus.UPLOADED)
    progress = Column(Float, default=0.0)  # 0.0 to 1.0
    current_stage = Column(String(100))
    error_message = Column(Text, nullable=True)

    # Job Information
    job_id = Column(String(100), nullable=True)  # Celery task ID
    processing_started_at = Column(DateTime, nullable=True)
    processing_completed_at = Column(DateTime, nullable=True)
    processing_time_seconds = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)


class SegmentationResult(Base):
    """Segmentation results for each case"""
    __tablename__ = "segmentation_results"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String(50), index=True, nullable=False)

    # Segmentation Files
    segmentation_path = Column(String(500))  # Path to segmentation mask .nii.gz
    zarr_path = Column(String(500), nullable=True)  # Compressed storage

    # Vessel Segmentation Metrics
    dice_score = Column(Float, nullable=True)
    iou_score = Column(Float, nullable=True)
    vessel_volume_mm3 = Column(Float, nullable=True)

    # Per-Vessel Results
    lmca_detected = Column(Integer, default=0)  # Boolean: 0 or 1
    lad_detected = Column(Integer, default=0)
    lcx_detected = Column(Integer, default=0)
    rca_detected = Column(Integer, default=0)

    # Model Information
    model_name = Column(String(200))
    model_version = Column(String(50))

    # Timestamps
    created_at = Column(DateTime, default=func.now())


class StenosisResult(Base):
    """Stenosis classification results per vessel"""
    __tablename__ = "stenosis_results"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String(50), index=True, nullable=False)

    # Vessel Information
    vessel_name = Column(String(50), nullable=False)  # LMCA, LAD, LCx, RCA
    vessel_segment = Column(String(50), nullable=True)  # proximal, mid, distal

    # Stenosis Measurements
    stenosis_percentage = Column(Float)  # 0.0 to 1.0 (0% to 100%)
    severity_class = Column(SQLEnum(StenosisSeverity))
    confidence_score = Column(Float)  # Model confidence 0.0 to 1.0

    # Location Information
    stenosis_location_mm = Column(Float, nullable=True)  # Distance from vessel origin

    # Radius Measurements
    reference_radius_mm = Column(Float, nullable=True)
    minimum_radius_mm = Column(Float, nullable=True)

    # CORAL Severity Matrix (stored as JSON)
    severity_probabilities = Column(JSON)  # {0: 0.1, 1: 0.3, 2: 0.5, 3: 0.1}

    # Centerline Data (optional, can be large)
    centerline_path = Column(String(500), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=func.now())


class ProcessingLog(Base):
    """Detailed processing logs for debugging"""
    __tablename__ = "processing_logs"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String(50), index=True, nullable=False)

    stage = Column(String(100))  # preprocessing, segmentation, classification
    message = Column(Text)
    log_level = Column(String(20))  # INFO, WARNING, ERROR
    timestamp = Column(DateTime, default=func.now())

    # Additional metadata
    metadata = Column(JSON, nullable=True)

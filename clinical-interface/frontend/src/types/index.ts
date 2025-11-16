/**
 * TypeScript type definitions for the clinical interface
 */

export enum CaseStatus {
  UPLOADED = 'uploaded',
  VALIDATING = 'validating',
  PREPROCESSING = 'preprocessing',
  SEGMENTING = 'segmenting',
  CLASSIFYING = 'classifying',
  COMPLETE = 'complete',
  ERROR = 'error',
  CANCELLED = 'cancelled',
}

export enum StenosisSeverity {
  NORMAL = 'normal',
  MILD = 'mild',
  MODERATE = 'moderate',
  SEVERE = 'severe',
}

export interface CaseListItem {
  case_id: string;
  original_filename: string;
  upload_date: string;
  status: CaseStatus;
  progress: number;
  dimensions?: number[];
  spacing?: number[];
}

export interface CaseDetail {
  case_id: string;
  original_filename: string;
  file_size_mb?: number;
  status: CaseStatus;
  progress: number;
  current_stage?: string;
  error_message?: string;
  dimensions?: number[];
  spacing?: number[];
  orientation?: string;
  scan_date?: string;
  job_id?: string;
  processing_started_at?: string;
  processing_completed_at?: string;
  processing_time_seconds?: number;
  created_at: string;
  updated_at: string;
}

export interface ProcessingStatus {
  case_id: string;
  status: CaseStatus;
  progress: number;
  current_stage: string;
  estimated_time_remaining_seconds?: number;
}

export interface VesselSegmentation {
  vessel_name: string;
  detected: boolean;
  volume_mm3?: number;
  confidence?: number;
}

export interface SegmentationResults {
  case_id: string;
  segmentation_path: string;
  zarr_path?: string;
  dice_score?: number;
  iou_score?: number;
  vessels: VesselSegmentation[];
  model_name: string;
  model_version: string;
}

export interface StenosisAnalysis {
  vessel_name: string;
  vessel_segment?: string;
  stenosis_percentage: number;
  severity_class: StenosisSeverity;
  confidence_score: number;
  stenosis_location_mm?: number;
  reference_radius_mm?: number;
  minimum_radius_mm?: number;
  severity_probabilities: Record<string, number>;
}

export interface StenosisResults {
  case_id: string;
  analyses: StenosisAnalysis[];
  overall_severity: StenosisSeverity;
  most_severe_vessel: string;
  most_severe_stenosis: number;
}

export interface CaseResults {
  case_id: string;
  status: CaseStatus;
  segmentation?: SegmentationResults;
  stenosis?: StenosisResults;
  processing_time_seconds?: number;
}

export interface VolumeMetadata {
  dimensions: number[];
  spacing: number[];
  origin: number[];
  orientation: string;
  num_slices: Record<string, number>;
}

export interface UploadResponse {
  case_id: string;
  status: CaseStatus;
  message: string;
  upload_info: {
    filename: string;
    size_mb: number;
    dimensions: number[];
    spacing: number[];
  };
}

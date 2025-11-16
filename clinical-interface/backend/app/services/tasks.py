"""
Celery Tasks
Async task processing for long-running operations
"""
from celery import Celery
from celery.utils.log import get_task_logger
from pathlib import Path
import asyncio
from typing import Dict, Optional

from app.core.config import settings
from app.services.inference import StenosisDetectionPipeline

logger = get_task_logger(__name__)

# Initialize Celery app
celery_app = Celery(
    "stenosis_detection",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

celery_app.conf.update(
    task_track_started=True,
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
    task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
    result_expires=3600,  # Results expire after 1 hour
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=10,
)


# Global pipeline instance (initialized once per worker)
_pipeline: Optional[StenosisDetectionPipeline] = None


def get_pipeline() -> StenosisDetectionPipeline:
    """Get or initialize the inference pipeline"""
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing inference pipeline in worker")
        _pipeline = StenosisDetectionPipeline(
            segmentation_model_path=settings.SEGMENTATION_MODEL_PATH,
            classification_model_path=settings.CLASSIFICATION_MODEL_PATH,
            device=settings.DEVICE
        )
    return _pipeline


@celery_app.task(bind=True, name="process_case")
def process_case_task(
    self,
    case_id: str,
    nifti_path: str,
    output_dir: str,
    config: Optional[Dict] = None
) -> Dict:
    """
    Process a single case through the complete pipeline

    Args:
        case_id: Unique case identifier
        nifti_path: Path to input NIFTI file
        output_dir: Directory for outputs
        config: Optional processing configuration

    Returns:
        Dictionary with processing results
    """
    logger.info(f"Starting task for case {case_id}")

    try:
        # Update task state
        self.update_state(
            state='PREPROCESSING',
            meta={'case_id': case_id, 'stage': 'preprocessing', 'progress': 0.1}
        )

        # Get pipeline
        pipeline = get_pipeline()

        # Run preprocessing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            preprocessed, metadata = loop.run_until_complete(
                pipeline.preprocess(
                    nifti_path,
                    output_dir,
                    hu_min=config.get("hu_min", -100) if config else -100,
                    hu_max=config.get("hu_max", 1000) if config else 1000,
                    target_spacing=config.get("target_spacing", 0.8) if config else 0.8
                )
            )

            # Update state
            self.update_state(
                state='SEGMENTING',
                meta={'case_id': case_id, 'stage': 'segmentation', 'progress': 0.3}
            )

            # Run segmentation
            if pipeline.seg_model is not None:
                segmentation = loop.run_until_complete(
                    pipeline.segment(preprocessed)
                )

                # Extract vessel masks
                vessel_masks = pipeline.extract_vessel_masks(segmentation)

                # Update state
                self.update_state(
                    state='CLASSIFYING',
                    meta={'case_id': case_id, 'stage': 'classification', 'progress': 0.7}
                )

                # Run classification
                stenosis_results = {}
                if pipeline.clf_model is not None:
                    stenosis_results = loop.run_until_complete(
                        pipeline.classify_stenosis(preprocessed, segmentation)
                    )

                # Compile results
                results = {
                    "case_id": case_id,
                    "status": "complete",
                    "preprocessing": metadata,
                    "segmentation": {
                        "vessels": {
                            name: {"detected": bool(mask.sum() > 100)}
                            for name, mask in vessel_masks.items()
                        }
                    },
                    "stenosis": stenosis_results
                }

            else:
                results = {
                    "case_id": case_id,
                    "status": "complete",
                    "preprocessing": metadata,
                    "error": "Segmentation model not loaded"
                }

            logger.info(f"Task completed for case {case_id}")
            return results

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"Task failed for case {case_id}: {str(e)}")
        self.update_state(
            state='FAILURE',
            meta={'case_id': case_id, 'error': str(e)}
        )
        raise


@celery_app.task(name="generate_report")
def generate_report_task(
    case_id: str,
    results: Dict,
    output_path: str,
    template: str = "comprehensive"
) -> str:
    """
    Generate PDF report for a case

    Args:
        case_id: Case identifier
        results: Processing results dictionary
        output_path: Output file path
        template: Report template ("brief" or "comprehensive")

    Returns:
        Path to generated report
    """
    logger.info(f"Generating {template} report for case {case_id}")

    try:
        # TODO: Implement PDF generation with reportlab
        # For now, just create a placeholder
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(f"Report for {case_id}\n\nResults: {results}")

        logger.info(f"Report generated: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        raise


@celery_app.task(name="cleanup_old_files")
def cleanup_old_files_task(days: int = 7) -> Dict:
    """
    Cleanup old processed files

    Args:
        days: Delete files older than this many days

    Returns:
        Dictionary with cleanup statistics
    """
    logger.info(f"Cleaning up files older than {days} days")

    import time
    from datetime import datetime, timedelta

    cutoff_time = time.time() - (days * 24 * 3600)
    deleted_count = 0
    freed_space_mb = 0

    try:
        for directory in [settings.PROCESSED_DIR, settings.REPORTS_DIR]:
            for file_path in Path(directory).rglob("*"):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    file_path.unlink()
                    deleted_count += 1
                    freed_space_mb += size_mb

        logger.info(f"Cleanup complete: {deleted_count} files, {freed_space_mb:.2f} MB freed")

        return {
            "deleted_files": deleted_count,
            "freed_space_mb": round(freed_space_mb, 2)
        }

    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        raise

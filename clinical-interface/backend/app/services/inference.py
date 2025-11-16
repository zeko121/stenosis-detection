"""
Inference Pipeline
Wrapper for stenosis detection models (segmentation + classification)
"""
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
import sys
import zarr

# Add parent directories to path to import existing modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from models.coronary_segmentation.src.unet import UNet3D
from src.preprocess import preprocess_volume

logger = logging.getLogger(__name__)


class StenosisDetectionPipeline:
    """
    Complete pipeline for stenosis detection
    Combines preprocessing, segmentation, and classification
    """

    def __init__(
        self,
        segmentation_model_path: Optional[str] = None,
        classification_model_path: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Initialize the pipeline

        Args:
            segmentation_model_path: Path to trained 3D U-Net checkpoint
            classification_model_path: Path to CORAL classification model
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing pipeline on device: {self.device}")

        # Load segmentation model
        self.seg_model = None
        if segmentation_model_path and Path(segmentation_model_path).exists():
            self.load_segmentation_model(segmentation_model_path)

        # Load classification model (TODO: implement when CORAL model is ready)
        self.clf_model = None
        if classification_model_path and Path(classification_model_path).exists():
            self.load_classification_model(classification_model_path)

    def load_segmentation_model(self, model_path: str):
        """Load pre-trained 3D U-Net model"""
        try:
            logger.info(f"Loading segmentation model from {model_path}")

            # Initialize model architecture
            self.seg_model = UNet3D(
                in_channels=1,
                out_channels=5,  # background + 4 vessels (LMCA, LAD, LCx, RCA)
                init_features=32
            ).to(self.device)

            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.seg_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.seg_model.load_state_dict(checkpoint)

            self.seg_model.eval()
            logger.info("Segmentation model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load segmentation model: {e}")
            raise

    def load_classification_model(self, model_path: str):
        """Load pre-trained CORAL classification model"""
        # TODO: Implement when CORAL model is available
        logger.warning("Classification model loading not yet implemented")
        pass

    async def preprocess(
        self,
        nifti_path: str,
        output_dir: str,
        hu_min: int = -100,
        hu_max: int = 1000,
        target_spacing: float = 0.8
    ) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess NIFTI volume

        Args:
            nifti_path: Path to input NIFTI file
            output_dir: Directory for preprocessed output
            hu_min: Minimum HU value for windowing
            hu_max: Maximum HU value for windowing
            target_spacing: Target isotropic spacing in mm

        Returns:
            Tuple of (preprocessed_array, metadata)
        """
        logger.info(f"Preprocessing {nifti_path}")

        try:
            # Load NIFTI
            nifti = nib.load(nifti_path)
            volume = nifti.get_fdata()
            affine = nifti.affine
            header = nifti.header
            original_spacing = header.get_zooms()

            logger.info(f"Original shape: {volume.shape}, spacing: {original_spacing}")

            # Use existing preprocessing function
            preprocessed = preprocess_volume(
                volume,
                original_spacing,
                target_spacing=(target_spacing, target_spacing, target_spacing),
                hu_window=(hu_min, hu_max),
                normalize=True
            )

            # Save preprocessed volume as Zarr for efficient storage
            output_path = Path(output_dir) / f"preprocessed_{Path(nifti_path).stem}.zarr"
            zarr.save(str(output_path), preprocessed)

            metadata = {
                "original_shape": list(volume.shape),
                "preprocessed_shape": list(preprocessed.shape),
                "original_spacing": list(original_spacing),
                "target_spacing": [target_spacing] * 3,
                "affine": affine.tolist(),
                "zarr_path": str(output_path)
            }

            logger.info(f"Preprocessing complete. Shape: {preprocessed.shape}")
            return preprocessed, metadata

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise

    async def segment(
        self,
        volume: np.ndarray,
        batch_size: int = 1
    ) -> np.ndarray:
        """
        Run segmentation inference

        Args:
            volume: Preprocessed 3D volume (normalized)
            batch_size: Batch size for inference

        Returns:
            Segmentation mask (H, W, D, C) with probabilities for each class
        """
        if self.seg_model is None:
            raise RuntimeError("Segmentation model not loaded")

        logger.info(f"Running segmentation inference on volume shape: {volume.shape}")

        try:
            with torch.no_grad():
                # Prepare input: add batch and channel dimensions
                # Input shape: (1, 1, D, H, W) for 3D U-Net
                input_tensor = torch.from_numpy(volume).float()

                # Handle different input shapes
                if input_tensor.ndim == 3:
                    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel
                elif input_tensor.ndim == 4:
                    input_tensor = input_tensor.unsqueeze(0)  # Add batch only

                input_tensor = input_tensor.to(self.device)

                logger.info(f"Input tensor shape: {input_tensor.shape}")

                # Run inference
                output = self.seg_model(input_tensor)

                # Apply softmax to get probabilities
                output = torch.softmax(output, dim=1)

                # Convert to numpy
                segmentation = output.cpu().numpy()

                # Remove batch dimension: (1, C, D, H, W) -> (C, D, H, W)
                segmentation = segmentation.squeeze(0)

                # Transpose to (D, H, W, C) for easier handling
                segmentation = np.transpose(segmentation, (1, 2, 3, 0))

                logger.info(f"Segmentation complete. Output shape: {segmentation.shape}")
                return segmentation

        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            raise

    def extract_vessel_masks(self, segmentation: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract individual vessel masks from multi-class segmentation

        Args:
            segmentation: Multi-class segmentation (D, H, W, C)

        Returns:
            Dictionary mapping vessel names to binary masks
        """
        vessel_names = ["background", "LMCA", "LAD", "LCx", "RCA"]
        masks = {}

        for i, name in enumerate(vessel_names[1:], start=1):  # Skip background
            # Get argmax to create hard segmentation
            hard_seg = np.argmax(segmentation, axis=-1)
            vessel_mask = (hard_seg == i).astype(np.uint8)
            masks[name] = vessel_mask

            volume = np.sum(vessel_mask)
            logger.info(f"{name} volume: {volume} voxels")

        return masks

    def calculate_dice_score(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate Dice similarity coefficient"""
        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target)

        if union == 0:
            return 1.0 if np.sum(pred) == 0 else 0.0

        dice = (2.0 * intersection) / union
        return float(dice)

    async def classify_stenosis(
        self,
        volume: np.ndarray,
        segmentation: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Classify stenosis severity for each vessel

        Args:
            volume: Preprocessed volume
            segmentation: Vessel segmentation masks

        Returns:
            Dictionary with stenosis analysis per vessel
        """
        # TODO: Implement CORAL classification when model is ready
        logger.warning("Stenosis classification not yet implemented - returning dummy data")

        # For now, return dummy data
        vessel_results = {}
        vessel_names = ["LMCA", "LAD", "LCx", "RCA"]

        for vessel in vessel_names:
            # Placeholder - will be replaced with actual CORAL inference
            vessel_results[vessel] = {
                "stenosis_percentage": 0.0,
                "severity_class": "normal",
                "confidence_score": 0.0,
                "severity_probabilities": {
                    "0": 1.0,
                    "1": 0.0,
                    "2": 0.0,
                    "3": 0.0
                }
            }

        return vessel_results

    async def process_case(
        self,
        nifti_path: str,
        output_dir: str,
        config: Optional[Dict] = None
    ) -> Dict:
        """
        Complete processing pipeline for a single case

        Args:
            nifti_path: Path to input NIFTI file
            output_dir: Directory for outputs
            config: Optional configuration overrides

        Returns:
            Dictionary with all results
        """
        logger.info(f"Starting complete processing for {nifti_path}")

        config = config or {}
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {
            "case_id": Path(nifti_path).stem,
            "input_path": nifti_path,
            "output_dir": output_dir
        }

        try:
            # Step 1: Preprocessing
            preprocessed, metadata = await self.preprocess(
                nifti_path,
                output_dir,
                hu_min=config.get("hu_min", -100),
                hu_max=config.get("hu_max", 1000),
                target_spacing=config.get("target_spacing", 0.8)
            )
            results["preprocessing"] = metadata

            # Step 2: Segmentation
            if self.seg_model is not None:
                segmentation = await self.segment(preprocessed)

                # Save segmentation
                seg_path = output_path / "segmentation.nii.gz"
                seg_nifti = nib.Nifti1Image(
                    np.argmax(segmentation, axis=-1).astype(np.uint8),
                    affine=np.array(metadata["affine"])
                )
                nib.save(seg_nifti, seg_path)

                # Extract vessel masks
                vessel_masks = self.extract_vessel_masks(segmentation)

                results["segmentation"] = {
                    "path": str(seg_path),
                    "vessels": {
                        name: {"detected": np.sum(mask) > 100}  # Threshold for detection
                        for name, mask in vessel_masks.items()
                    }
                }

                # Step 3: Classification (if model available)
                if self.clf_model is not None:
                    stenosis_results = await self.classify_stenosis(preprocessed, segmentation)
                    results["stenosis"] = stenosis_results

            logger.info("Processing complete")
            return results

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            results["error"] = str(e)
            raise

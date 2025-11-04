# AI Agent Instructions for Stenosis Detection Project

## Project Overview

This is a Python-based medical imaging project that performs automatic detection and assessment of coronary artery stenosis from Coronary CT Angiography (CCTA) scans. The project follows a multi-stage pipeline architecture.

## Key Components

1. **Preprocessing Pipeline** (`src/preprocess.py`)
   - Handles NIFTI format medical images
   - Key operations: HU value clipping (-100 to 1000), resampling to 1mm isotropic, normalization to [0,1]
   - Maintains geometric metadata (affine matrices, spacing)
   - Verifies data integrity with extensive checks
   - Logs processing statistics to `logs/preprocess_log.csv`

2. **Visualization Tools** (`scripts/visualize_preprocessing.py`)
   - Displays slice comparisons across axial/coronal/sagittal planes 
   - Generates intensity distribution histograms
   - Supports multiple histogram visualization modes (split, overlay, dual-axis)

3. **Model Components** (under development)
   - Coronary Segmentation (`models/coronary_segmentation/`): U-Net based architecture
   - Stenosis Labeling (`models/stenosis_labeling/`): CORAL ordinal regression

## Directory Structure

```
data/
  imageCAS/          # Preprocessed data
  imageCAS_data/     # Raw CCTA scans (1-1000)
  processed/         # Preprocessed outputs
logs/                # Processing logs
models/             
  coronary_segmentation/  # Segmentation model
  stenosis_labeling/      # Classification model
outputs/
  viz_compare/      # Visualization outputs
scripts/            # Analysis scripts
src/                # Core processing code
```

## Development Patterns

1. **Data Processing**
   - Raw scans are in `imageCAS_data/` split into 200-case chunks
   - Use `preprocess.py` for initial data preparation
   - Always verify outputs with `visualize_preprocessing.py`
   - Check `logs/preprocess_log.csv` for processing stats

2. **Code Organization**
   - Core functionality in `src/`
   - Visualization/analysis tools in `scripts/`
   - Model-specific code separated into respective folders

3. **File Handling**
   - Medical image data uses NIFTI format (.nii/.nii.gz)
   - Binary masks use `.label.` or `_mask` in filename
   - Results/artifacts stored in `outputs/` directory

## Important Notes

1. Large medical imaging data is excluded from git (.gitignore)
2. Model checkpoints and training runs are stored locally
3. All preprocessing maintains proper geometric metadata
4. Extensive validation is performed during preprocessing

## Common Tasks

1. **Preprocessing Data**:
   ```python
   python src/preprocess.py  # Processes first N cases
   ```

2. **Visualizing Results**:
   ```python
   python scripts/visualize_preprocessing.py  # Shows comparison plots
   ```
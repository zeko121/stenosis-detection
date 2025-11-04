# Coronary Artery Stenosis Detection from CCTA

This project aims to develop an AI-driven pipeline for **automatic detection and assessment of coronary artery stenosis** from **Coronary CT Angiography (CCTA)** scans.  
It is part of the final capstone project at the **University of Haifa (Information Systems Department)** in collaboration with **Ziv Medical Center**.

---

## Project Overview

This system consists of two main machine learning components:

 **1. Coronary Segmentation Head**  Localizes and segments major coronary arteries (e.g., LMCA, LAD, RCA) using 3D volumetric CCTA scans.
 **2. Stenosis Severity Labeling Head** Predicts stenosis severity for each artery segment based on segmentation output and features.  Ordinal classification (e.g., No / Mild / Moderate / Severe). |

The project follows a **multi-stage pipeline**:
1. **Data preprocessing** (resampling, HU clipping, normalization).
2. **Artery segmentation** (U-Net / MONAI-based models).
3. **ROI-based feature extraction / pooling**.
4. **Stenosis severity classification** (CNN / CORAL / Ordinal regression).
5. **Evaluation & visualization**.
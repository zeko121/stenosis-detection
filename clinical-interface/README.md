# Clinical Stenosis Detection Interface

A professional web-based clinical interface for AI-powered coronary artery stenosis detection from Cardiac CT Angiography (CCTA) scans.

**Project:** University of Haifa (Information Systems Department) × Ziv Medical Center
**Supervisor:** Prof. Mario Boley
**Status:** Research Prototype

---

## Overview

This clinical interface provides a complete workflow for:
- **Upload** NIFTI format CCTA scans
- **Process** scans through 3D U-Net segmentation and CORAL stenosis classification
- **Visualize** multi-planar reconstructions (MPR) with segmentation overlays
- **Analyze** stenosis severity per coronary artery (LMCA, LAD, LCx, RCA)
- **Generate** clinical-grade PDF reports

## Architecture

```
clinical-interface/
├── backend/              # FastAPI backend
│   ├── app/
│   │   ├── api/         # REST API endpoints
│   │   ├── core/        # Configuration & database
│   │   ├── models/      # SQLAlchemy models & Pydantic schemas
│   │   ├── services/    # Inference pipeline & Celery tasks
│   │   └── main.py      # FastAPI application
│   └── requirements.txt
├── frontend/            # React + TypeScript frontend
│   ├── src/
│   │   ├── api/        # API client
│   │   ├── components/ # React components
│   │   ├── types/      # TypeScript definitions
│   │   └── App.tsx
│   └── package.json
├── docker/              # Docker configuration
│   ├── docker-compose.yml
│   ├── Dockerfile.backend
│   └── Dockerfile.frontend
└── data/               # Runtime data storage
    ├── uploads/        # Uploaded NIFTI files
    ├── processed/      # Segmentation results
    └── reports/        # Generated PDF reports
```

## Technology Stack

### Backend
- **Framework:** FastAPI (async Python web framework)
- **Deep Learning:** PyTorch 2.0+ with CUDA support
- **Medical Imaging:** MONAI, NiBabel, SimpleITK
- **Task Queue:** Celery + Redis (async processing)
- **Database:** SQLite (async with aiosqlite)
- **Compression:** Zarr with Blosc codec

### Frontend
- **Framework:** React 18 + TypeScript
- **Build Tool:** Vite
- **Styling:** TailwindCSS
- **Data Fetching:** React Query (TanStack Query)
- **Routing:** React Router v6
- **File Upload:** react-dropzone
- **Charts:** Recharts

## Quick Start

### Prerequisites

- **Docker & Docker Compose** (recommended)
- **OR Manual Setup:**
  - Python 3.10+
  - Node.js 18+
  - Redis server
  - CUDA-capable GPU (optional but recommended)

### Option 1: Docker (Recommended)

```bash
# Navigate to docker directory
cd clinical-interface/docker

# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f

# Access application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Option 2: Manual Setup

**Backend:**

```bash
cd clinical-interface/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env to set model paths and configuration

# Start Redis (in separate terminal)
redis-server

# Start Celery worker (in separate terminal)
celery -A app.services.tasks worker --loglevel=info

# Start FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**

```bash
cd clinical-interface/frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Access at http://localhost:3000
```

## Configuration

### Backend Environment Variables

Create `backend/.env`:

```env
# API Configuration
API_V1_STR=/api/v1
DEBUG=False

# Database
DATABASE_URL=sqlite+aiosqlite:///./stenosis_clinical.db

# Redis & Celery
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Model Paths (IMPORTANT: Update these!)
SEGMENTATION_MODEL_PATH=../../models/coronary_segmentation/checkpoints/best_model.pth
CLASSIFICATION_MODEL_PATH=../../models/coral_classification/checkpoints/best_model.pth

# Processing
DEVICE=cuda  # or 'cpu'
MAX_UPLOAD_SIZE_MB=500
HU_WINDOW_MIN=-100
HU_WINDOW_MAX=1000
TARGET_SPACING=0.8

# Security (CHANGE IN PRODUCTION!)
SECRET_KEY=YOUR_RANDOM_SECRET_KEY_HERE
```

**Generate a secure secret key:**
```bash
openssl rand -hex 32
```

### Frontend Configuration

Create `frontend/.env`:

```env
VITE_API_URL=http://localhost:8000
```

## Usage Guide

### 1. Upload a CCTA Scan

1. Navigate to **Upload** page
2. Drag & drop a NIFTI file (`.nii` or `.nii.gz`)
3. System validates file and extracts metadata
4. Click "View Case Details" to proceed

### 2. Process the Case

1. From the case detail page, click "Start Processing"
2. Monitor real-time progress:
   - Preprocessing (10%)
   - Segmentation (30-70%)
   - Classification (70-90%)
   - Report Generation (90-100%)

### 3. Review Results

Once processing is complete, view:

- **Stenosis Analysis Cards:** Per-vessel severity with confidence scores
- **Vessel Segmentation:** Detection status for LMCA, LAD, LCx, RCA
- **Summary Statistics:** Most severe vessel, max stenosis percentage
- **Metadata:** Scan dimensions, spacing, processing time

### 4. Generate Report

Click "Download Report" to generate a comprehensive PDF with:
- Patient/scan metadata
- Stenosis findings table
- Key visualizations
- Clinical interpretation

## API Documentation

### Key Endpoints

**Case Management:**
- `POST /api/v1/cases/upload` - Upload NIFTI scan
- `GET /api/v1/cases` - List all cases
- `GET /api/v1/cases/{case_id}` - Get case details
- `POST /api/v1/cases/{case_id}/process` - Start processing
- `GET /api/v1/cases/{case_id}/status` - Check processing status
- `DELETE /api/v1/cases/{case_id}` - Delete case

**Results:**
- `GET /api/v1/results/{case_id}` - Get analysis results
- `POST /api/v1/results/{case_id}/report` - Generate report

**Visualization:**
- `GET /api/v1/visualization/{case_id}/metadata` - Volume metadata
- `GET /api/v1/visualization/{case_id}/slice` - Get 2D slice
- `GET /api/v1/visualization/{case_id}/volume_data` - 3D volume data

**Interactive API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Development

### Backend Testing

```bash
cd backend

# Install dev dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

### Frontend Testing

```bash
cd frontend

# Install dev dependencies
npm install --save-dev @testing-library/react @testing-library/jest-dom vitest

# Run tests
npm run test
```

### Code Quality

**Backend:**
```bash
# Format code
black app/

# Lint
flake8 app/

# Type checking
mypy app/
```

**Frontend:**
```bash
# Lint
npm run lint

# Type checking
npx tsc --noEmit
```

## Deployment

### Production Checklist

- [ ] Change `SECRET_KEY` in `.env`
- [ ] Set `DEBUG=False`
- [ ] Configure proper CORS origins
- [ ] Set up SSL/TLS certificates (HTTPS)
- [ ] Configure production database (PostgreSQL recommended)
- [ ] Set up proper logging and monitoring
- [ ] Configure backup strategy for data
- [ ] Review and harden security settings
- [ ] Set up proper authentication/authorization
- [ ] Configure resource limits (CPU, memory, GPU)

### Docker Production Deployment

```bash
# Build for production
docker-compose -f docker-compose.prod.yml build

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Check health
docker-compose ps
docker-compose logs
```

## Troubleshooting

### Issue: CUDA not available

**Solution:**
- Ensure NVIDIA Docker runtime is installed
- Check GPU availability: `nvidia-smi`
- Update `DEVICE=cpu` in `.env` if no GPU

### Issue: Celery worker not processing tasks

**Solution:**
- Check Redis connection: `redis-cli ping`
- Verify Celery worker is running: `docker-compose logs celery-worker`
- Restart worker: `docker-compose restart celery-worker`

### Issue: Upload fails with "File too large"

**Solution:**
- Increase `MAX_UPLOAD_SIZE_MB` in backend `.env`
- Update nginx/proxy max body size if applicable

### Issue: Frontend cannot connect to backend

**Solution:**
- Check backend is running: `curl http://localhost:8000/health`
- Verify CORS origins in backend config
- Check proxy configuration in `vite.config.ts`

## Model Integration

### Adding New Models

1. **Train your model** using the existing training scripts
2. **Save checkpoint** in `.pth` format
3. **Update configuration:**
   ```env
   SEGMENTATION_MODEL_PATH=/path/to/new_segmentation_model.pth
   CLASSIFICATION_MODEL_PATH=/path/to/new_classification_model.pth
   ```
4. **Restart services:**
   ```bash
   docker-compose restart backend celery-worker
   ```

### Model Requirements

**Segmentation Model:**
- Architecture: 3D U-Net compatible with `models/coronary_segmentation/src/unet.py`
- Input: `(B, 1, D, H, W)` - normalized CT volume
- Output: `(B, 5, D, H, W)` - 5-class probabilities (background + 4 vessels)

**Classification Model (CORAL):**
- Architecture: Ordinal regression network
- Input: Preprocessed volume + segmentation mask
- Output: Severity probabilities for each vessel

## Performance Benchmarks

### Processing Time (GPU: NVIDIA RTX 3090)

| Step | Time (avg) | Memory |
|------|-----------|--------|
| Preprocessing | 10-15s | 2GB |
| Segmentation | 30-45s | 6GB |
| Classification | 5-10s | 4GB |
| **Total** | **45-70s** | **8GB** |

### Processing Time (CPU: 16-core)

| Step | Time (avg) | Memory |
|------|-----------|--------|
| Preprocessing | 30-45s | 4GB |
| Segmentation | 3-5 min | 8GB |
| Classification | 20-30s | 6GB |
| **Total** | **4-6 min** | **12GB** |

## Security Considerations

⚠️ **IMPORTANT:** This is a research prototype. For clinical deployment:

1. **Authentication:** Implement proper user authentication (OAuth2, SAML)
2. **Authorization:** Role-based access control (RBAC)
3. **Data Encryption:** Encrypt data at rest and in transit
4. **Audit Logging:** Track all access and modifications
5. **HIPAA Compliance:** Ensure compliance with healthcare data regulations
6. **PHI Handling:** Implement proper de-identification procedures
7. **Network Security:** Deploy behind firewall, use VPN for remote access

## License

This project is developed for research purposes at University of Haifa in collaboration with Ziv Medical Center.

**Disclaimer:** This AI-assisted stenosis detection system is for research purposes only and is NOT FDA-approved for clinical diagnosis. All results must be validated by a qualified radiologist or cardiologist before clinical use.

## Support & Contact

- **Issues:** Report bugs via GitHub Issues
- **Supervisor:** Prof. Mario Boley (University of Haifa)
- **Medical Partner:** Ziv Medical Center

## Acknowledgments

- **University of Haifa** - Information Systems Department
- **Ziv Medical Center** - Clinical collaboration and validation
- **Prof. Mario Boley** - Project supervision

## Citation

If you use this system in your research, please cite:

```bibtex
@software{stenosis_detection_interface,
  title={Clinical Interface for AI-Powered Coronary Stenosis Detection},
  author={University of Haifa Information Systems Department},
  year={2025},
  institution={University of Haifa, Ziv Medical Center},
  supervisor={Mario Boley}
}
```

---

**Last Updated:** November 2025
**Version:** 1.0.0

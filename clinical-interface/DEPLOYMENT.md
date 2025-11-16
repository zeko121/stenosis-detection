# Deployment Guide - Clinical Stenosis Detection Interface

This guide covers deployment options for the clinical stenosis detection system.

## Table of Contents

1. [Development Deployment](#development-deployment)
2. [Production Deployment](#production-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Performance Optimization](#performance-optimization)
5. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Development Deployment

### Local Development Setup

**Prerequisites:**
- Python 3.10+
- Node.js 18+
- Redis
- CUDA GPU (optional)

**Step 1: Clone Repository**
```bash
git clone <repository-url>
cd stenosis-detection/clinical-interface
```

**Step 2: Backend Setup**
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Initialize database
python -c "from app.core.database import init_db; import asyncio; asyncio.run(init_db())"
```

**Step 3: Start Redis**
```bash
# Option 1: Docker
docker run -d -p 6379:6379 redis:7-alpine

# Option 2: System service
sudo systemctl start redis
```

**Step 4: Start Backend Services**
```bash
# Terminal 1: FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Celery worker
celery -A app.services.tasks worker --loglevel=info --concurrency=2
```

**Step 5: Frontend Setup**
```bash
cd ../frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

**Access:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## Production Deployment

### Docker Compose Deployment (Recommended)

**Prerequisites:**
- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Docker (for GPU support)

**Step 1: Prepare Environment**
```bash
cd clinical-interface/docker

# Copy and edit environment files
cp ../backend/.env.example ../backend/.env
# Configure production settings
```

**Important Configuration Changes:**
```env
# backend/.env
DEBUG=False
SECRET_KEY=<generate-with-openssl-rand-hex-32>
DATABASE_URL=postgresql+asyncpg://user:pass@db:5432/stenosis
BACKEND_CORS_ORIGINS=["https://yourdomain.com"]
```

**Step 2: Build Images**
```bash
docker-compose build
```

**Step 3: Start Services**
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

**Step 4: Initialize Database**
```bash
docker-compose exec backend python -c "from app.core.database import init_db; import asyncio; asyncio.run(init_db())"
```

**Step 5: Configure Nginx Reverse Proxy**

Create `/etc/nginx/sites-available/stenosis`:
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    # SSL certificates (use Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Increase timeouts for long-running inference
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;

        # Increase max body size for file uploads
        client_max_body_size 500M;
    }
}
```

Enable and restart:
```bash
sudo ln -s /etc/nginx/sites-available/stenosis /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

**Step 6: Set Up SSL with Let's Encrypt**
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

---

## Cloud Deployment

### AWS Deployment

**Architecture:**
- EC2 (p3.2xlarge for GPU) or ECS with GPU support
- RDS PostgreSQL for database
- ElastiCache Redis for Celery
- S3 for data storage
- CloudFront for CDN

**Step 1: Launch EC2 Instance**
```bash
# Choose AMI: Deep Learning AMI (Ubuntu)
# Instance Type: p3.2xlarge (GPU) or t3.2xlarge (CPU)
# Storage: 100GB SSD minimum
```

**Step 2: Install Docker**
```bash
ssh ubuntu@<instance-ip>

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**Step 3: Clone and Deploy**
```bash
git clone <repository-url>
cd stenosis-detection/clinical-interface

# Configure for AWS
# - Update DATABASE_URL to RDS endpoint
# - Update REDIS_URL to ElastiCache endpoint
# - Configure S3 for data storage

docker-compose up -d
```

**Step 4: Configure Auto-Scaling (Optional)**
- Set up Auto Scaling Group
- Configure health checks
- Set scaling policies based on CPU/GPU utilization

### Google Cloud Platform (GCP)

**Architecture:**
- Compute Engine with GPU or GKE
- Cloud SQL for PostgreSQL
- Memorystore for Redis
- Cloud Storage for data

**Step 1: Create VM with GPU**
```bash
gcloud compute instances create stenosis-detection \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --zone=us-central1-a
```

**Step 2: SSH and Deploy**
```bash
gcloud compute ssh stenosis-detection

# Deploy using Docker Compose
# Configure Cloud SQL and Memorystore endpoints
```

### Azure Deployment

**Architecture:**
- Azure Container Instances or AKS with GPU
- Azure Database for PostgreSQL
- Azure Cache for Redis
- Azure Blob Storage

---

## Performance Optimization

### Backend Optimization

**1. Enable Model Caching**
```python
# In app/services/inference.py
# Models are already cached globally per worker
```

**2. Optimize Batch Processing**
```python
# Increase batch size if GPU memory allows
BATCH_SIZE=4  # in .env
```

**3. Use GPU Inference**
```env
DEVICE=cuda
ENABLE_MIXED_PRECISION=True
```

**4. Configure Celery Workers**
```bash
# CPU-bound: Set workers = CPU cores
celery -A app.services.tasks worker --concurrency=8

# GPU-bound: Set workers = 1-2 per GPU
celery -A app.services.tasks worker --concurrency=2 --pool=solo
```

### Frontend Optimization

**1. Build for Production**
```bash
npm run build
# Use serve or nginx to serve static files
```

**2. Enable Compression**
```nginx
# In nginx.conf
gzip on;
gzip_types text/plain text/css application/json application/javascript text/xml application/xml;
gzip_comp_level 6;
```

**3. Configure CDN**
- Use CloudFront (AWS) or Cloud CDN (GCP)
- Cache static assets
- Enable HTTP/2

### Database Optimization

**1. Migrate to PostgreSQL**
```env
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/stenosis
```

**2. Add Indexes**
```sql
CREATE INDEX idx_case_status ON cases(status);
CREATE INDEX idx_case_created ON cases(created_at DESC);
CREATE INDEX idx_segmentation_case ON segmentation_results(case_id);
```

**3. Configure Connection Pooling**
```python
# In app/core/database.py
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True
)
```

---

## Monitoring & Maintenance

### Logging

**Backend Logging:**
```python
# Configure structured logging
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'handlers': {
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/app/logs/stenosis.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'detailed',
        },
    },
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['file'],
    },
}
```

**Celery Logging:**
```bash
celery -A app.services.tasks worker --loglevel=info --logfile=/app/logs/celery.log
```

### Health Checks

**Endpoint Monitoring:**
```bash
# Health check script
#!/bin/bash
curl -f http://localhost:8000/health || exit 1
```

**Docker Health Checks:**
Already configured in Dockerfiles:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

### Backup Strategy

**Database Backups:**
```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
docker-compose exec -T backend sqlite3 /app/data/stenosis_clinical.db ".backup '/app/data/backups/backup_${DATE}.db'"

# For PostgreSQL
pg_dump -h localhost -U stenosis_user stenosis_db > backup_${DATE}.sql
```

**Data Backups:**
```bash
# Backup uploaded files and results
tar -czf data_backup_${DATE}.tar.gz clinical-interface/data/
```

**Automated Backups:**
```bash
# Add to crontab
0 2 * * * /path/to/backup_script.sh
```

### Monitoring Tools

**Prometheus + Grafana:**
```yaml
# docker-compose.monitoring.yml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

**Application Metrics:**
```python
# Add to FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

### Maintenance Tasks

**Weekly Tasks:**
- Review error logs
- Check disk space
- Verify backup integrity
- Monitor GPU utilization

**Monthly Tasks:**
- Update dependencies (security patches)
- Review and archive old cases
- Performance benchmarking
- Model performance evaluation

**Quarterly Tasks:**
- Full system audit
- Security review
- Model retraining with new data
- User feedback analysis

---

## Troubleshooting

### Common Issues

**1. Out of GPU Memory**
```bash
# Solution: Reduce batch size or use CPU
DEVICE=cpu  # in .env
```

**2. Celery Tasks Stuck**
```bash
# Purge queue and restart
celery -A app.services.tasks purge
docker-compose restart celery-worker
```

**3. Database Locked**
```bash
# Migrate to PostgreSQL for production
# SQLite doesn't handle concurrent writes well
```

**4. Slow Processing**
```bash
# Check GPU utilization
nvidia-smi

# Check Celery workers
celery -A app.services.tasks inspect active
```

---

## Security Checklist

Production deployment must include:

- [ ] HTTPS enabled (SSL/TLS certificates)
- [ ] Strong SECRET_KEY configured
- [ ] DEBUG=False in production
- [ ] CORS properly configured
- [ ] Database credentials secured
- [ ] File upload validation
- [ ] Rate limiting configured
- [ ] Authentication implemented
- [ ] Regular security updates
- [ ] Backup strategy in place
- [ ] Monitoring and alerting configured
- [ ] HIPAA compliance reviewed (if applicable)

---

## Support

For deployment issues:
1. Check logs: `docker-compose logs -f`
2. Review this guide
3. Contact system administrator

**Emergency Contacts:**
- Technical Lead: [contact]
- DevOps: [contact]
- Prof. Mario Boley (Supervisor)

---

**Last Updated:** November 2025

📄 `docs/deployment_guide.md`

```markdown
# 🚀 Deployment Guide

## Overview
This guide covers various deployment options for MedPredict AI, from local development to production cloud deployment.

## Quick Deployment

### Local Development
```bash
# 1. Clone repository
git clone https://github.com/Iceyma02/Disease-Prediction-Diagnosis-using-Machine-Learning.git
cd Disease-Prediction-Diagnosis-using-Machine-Learning

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run application
streamlit run streamlit_app/app.py

# 4. Access application
# Open http://localhost:8501 in your browser
```

## Deployment Options

### 1. Local Machine Deployment

#### Using Python Virtual Environment
```bash
# Create virtual environment
python -m venv medpredict_env
source medpredict_env/bin/activate  # Linux/Mac
medpredict_env\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run streamlit_app/app.py
```

#### Using Conda
```bash
# Create conda environment
conda env create -f environment.yml
conda activate disease-prediction

# Run application
streamlit run streamlit_app/app.py
```

### 2. Docker Deployment

#### Build and Run
```bash
# Build Docker image
docker build -t medpredict-ai .

# Run container
docker run -p 8501:8501 medpredict-ai

# Run with custom port
docker run -p 8080:8501 medpredict-ai
```

#### Docker Compose
Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  medpredict-ai:
    build: .
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

### 3. Cloud Deployment

#### Heroku
```bash
# Create Heroku app
heroku create your-medpredict-app

# Set buildpacks
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# Scale (if needed)
heroku ps:scale web=1
```

Required files for Heroku:

**Procfile**
```
web: sh setup.sh && streamlit run streamlit_app/app.py
```

**setup.sh**
```bash
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

#### AWS EC2 Deployment

```bash
# Connect to EC2 instance
ssh -i "your-key.pem" ubuntu@your-ec2-address

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3-pip -y

# Clone repository
git clone https://github.com/Iceyma02/Disease-Prediction-Diagnosis-using-Machine-Learning.git
cd Disease-Prediction-Diagnosis-using-Machine-Learning

# Install dependencies
pip3 install -r requirements.txt

# Run with nohup for background execution
nohup streamlit run streamlit_app/app.py --server.port 8501 --server.address 0.0.0.0 &
```

#### AWS Elastic Beanstalk

Create `.ebextensions/python.config`:
```yaml
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: streamlit_app:app
```

Deploy:
```bash
eb init medpredict-ai -p python-3.9
eb create medpredict-prod
eb deploy
```

### 4. Container Orchestration

#### Kubernetes Deployment

**medpredict-deployment.yaml**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: medpredict-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: medpredict-ai
  template:
    metadata:
      labels:
        app: medpredict-ai
    spec:
      containers:
      - name: medpredict-ai
        image: medpredict-ai:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: medpredict-service
spec:
  selector:
    app: medpredict-ai
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8501
  type: LoadBalancer
```

## Environment Configuration

### Environment Variables
```bash
# Application settings
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Model paths (if custom)
export HEART_MODEL_PATH=/app/models/heart_disease_model.pkl
export DIABETES_MODEL_PATH=/app/models/diabetes_model.pkl
```

### Configuration Files

**config.yaml**
```yaml
app:
  name: "MedPredict AI"
  version: "1.0.0"
  debug: false

server:
  port: 8501
  address: "0.0.0.0"
  
models:
  heart_disease: "models/heart_disease_model.pkl"
  diabetes: "models/diabetes_model.pkl"
  
data:
  heart_data: "data/raw/heart.csv"
  diabetes_data: "data/raw/diabetes.csv"
```

## Performance Optimization

### Resource Requirements
- **Minimum**: 1GB RAM, 1 CPU core
- **Recommended**: 2GB RAM, 2 CPU cores
- **Production**: 4GB RAM, 4 CPU cores

### Caching Strategies
```python
@st.cache_resource
def load_models():
    return DiseasePredictor()

@st.cache_data
def preprocess_data(data):
    return processor.prepare_features(data)
```

### Database Integration
```python
# Example with PostgreSQL
import psycopg2
import pandas as pd

def save_prediction(patient_data, prediction_result):
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO predictions (patient_data, prediction, probability, timestamp)
        VALUES (%s, %s, %s, NOW())
    """, (json.dumps(patient_data), prediction_result['prediction'], prediction_result['probability']))
    
    conn.commit()
    conn.close()
```

## Security Considerations

### Environment Security
```bash
# Secure environment variables
export DATABASE_URL="postgresql://user:pass@host:port/db"
export SECRET_KEY="your-secret-key"

# File permissions
chmod 600 config.yaml
chmod 700 models/
```

### HTTPS Configuration
```python
# Streamlit config for HTTPS
[server]
sslCertFile = "/path/to/cert.pem"
sslKeyFile = "/path/to/key.pem"
```

### Authentication (Optional)
```python
# Basic auth example
import streamlit_authenticator as stauth

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    
    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True
```

## Monitoring and Logging

### Application Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medpredict.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### Health Checks
```python
# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })
```

### Performance Monitoring
```python
import time
from prometheus_client import Counter, Histogram

# Metrics
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions')
PREDICTION_DURATION = Histogram('prediction_duration_seconds', 'Prediction duration')

def predict_with_metrics(features):
    start_time = time.time()
    result = predictor.predict(features)
    duration = time.time() - start_time
    
    PREDICTION_COUNT.inc()
    PREDICTION_DURATION.observe(duration)
    
    return result
```

## Backup and Recovery

### Model Backup
```bash
# Backup models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Restore models
tar -xzf models_backup_20231201.tar.gz
```

### Database Backup
```sql
-- PostgreSQL backup
pg_dump -h localhost -U username dbname > backup_$(date +%Y%m%d).sql

-- Restore
psql -h localhost -U username dbname < backup_20231201.sql
```

## Troubleshooting

### Common Issues

1. **Port already in use**
```bash
streamlit run streamlit_app/app.py --server.port 8502
```

2. **Model loading errors**
```bash
# Check model files
ls -la models/

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

3. **Memory issues**
```python
# Clear cache
st.cache_resource.clear()
st.cache_data.clear()
```

### Log Analysis
```bash
# View application logs
tail -f medpredict.log

# Docker logs
docker logs medpredict-container

# System resources
htop
free -h
```

## Support

For deployment issues:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure proper file permissions
4. Check system resource availability

Contact support through GitHub issues for additional help.

# ML Models Deployment with Docker & GitHub Actions

This repository contains three machine learning models deployed using Docker and GitHub Actions CI/CD.

## Models

1. **Linear Regression** (`linear.py`) - Predicts exam scores based on study hours
2. **Logistic Regression** (`logistic.py`) - Predicts customer sentiment based on e-commerce behavior
3. **Support Vector Machine** (`support_vector.py`) - Predicts telecom customer churn

## Prerequisites

- Docker installed locally
- Docker Hub account
- GitHub repository with Actions enabled
- Git configured on your machine

## Setup Instructions

### 1. Local Docker Setup

**Build the Docker image locally:**
```bash
docker build -t ml-models:latest .
```

**Run the Docker container:**
```bash
docker run -it ml-models:latest
```

### 2. GitHub Actions Setup

Add the following secrets to your GitHub repository:

**Go to:** Settings → Secrets and variables → Actions → New repository secret

- `DOCKER_USERNAME`: Your Docker Hub username
- `DOCKER_PASSWORD`: Your Docker Hub access token (not password)

### 3. Deploy to Docker Hub

**Push your code to GitHub:**
```bash
git add .
git commit -m "Add Docker and GitHub Actions CI/CD"
git push origin main
```

The GitHub Actions workflow will automatically:
1. Build the Docker image
2. Push to Docker Hub
3. Run tests

### 4. Pull and Run from Docker Hub

```bash
docker pull DOCKER_USERNAME/ml-models:latest
docker run -it DOCKER_USERNAME/ml-models:latest
```

## Workflow Details

The CI/CD pipeline (`docker-deploy.yml`) includes:

- **Build & Push**: Builds Docker image and pushes to Docker Hub
- **Test**: Runs all models with sample inputs
- **Automated on**: Push to main branch or pull requests

## Docker Commands

```bash
# Build image
docker build -t ml-models .

# Run container
docker run -it ml-models

# View images
docker images

# Remove image
docker rmi ml-models

# Push to registry
docker push username/ml-models:latest
```

## File Structure

```
.
├── linear.py
├── logistic.py
├── support_vector.py
├── data/
│   ├── Ecom_cust_sentiment_data.csv
│   ├── Exam_score_data.csv
│   └── Telecome_churn_data.csv
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .github/
│   └── workflows/
│       └── docker-deploy.yml
└── README.md
```

## Troubleshooting

**Docker Hub authentication fails:**
- Verify Docker credentials in GitHub Secrets
- Use Docker Hub access token, not password
- Check username matches Docker Hub account

**Container exits immediately:**
- Models expect user input; modify scripts for batch processing
- Or run interactively: `docker run -it image_name`

**Models fail in container:**
- Verify all dependencies in requirements.txt
- Check data files are included in COPY command
- View logs: `docker logs container_id`

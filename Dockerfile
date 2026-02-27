# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Copy data files
COPY data/ /app/data/

# Copy model scripts
COPY linear.py logistic.py support_vector.py /app/

# Copy requirements
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a simple API server to serve models (optional)
# If you want to run scripts directly, uncomment below
# CMD ["python", "linear.py"]

# For an interactive container with all models
CMD ["/bin/bash"]

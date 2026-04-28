FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Copy data and models from project root
# (mounted as volumes in docker-compose)

# Default: run preprocessing then training
CMD ["sh", "-c", "python data_preprocessing.py && python model_training.py"]
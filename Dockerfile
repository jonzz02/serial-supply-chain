# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create results directory
RUN mkdir -p results_master

# Default command (can be overridden)
# Run master_experiment.py with default arguments
# User can override with docker run command
CMD ["python", "master_experiment.py", "--n_seeds", "100", "--rounds", "365", "--warmup", "50", "--max_workers", "4", "--output_dir", "results_master"]

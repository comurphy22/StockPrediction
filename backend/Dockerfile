# Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"

# Copy backend code
COPY app/ ./app/

# Copy the src directory from parent (ML code)
COPY ../src/ ./src/

# Copy models directory
COPY ../models/ ./models/

# Expose port (Railway will use $PORT)
EXPOSE 8000

# Set Python path for imports
ENV PYTHONPATH=/app/src:/app

# Run the server with dynamic port from Railway
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]


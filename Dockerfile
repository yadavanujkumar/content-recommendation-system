# Docker setup for the Content Recommendation System

FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Generate data if not exists
RUN python generate_simple_data.py

# Expose ports
EXPOSE 8000 8080

# Create startup script
RUN echo '#!/bin/bash\npython src/api/main.py &\npython simple_frontend.py' > start.sh
RUN chmod +x start.sh

CMD ["./start.sh"]
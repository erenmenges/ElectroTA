FROM python:3.10-slim

# System libraries required by opencv-python
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements_server.txt .
RUN pip install --no-cache-dir -r requirements_server.txt

COPY server.py .
COPY breadboard_analysis/ ./breadboard_analysis/

CMD ["python", "server.py"]

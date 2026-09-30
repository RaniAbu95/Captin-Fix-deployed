FROM python:3.11-slim

# Install Chromium + chromedriver from Debian (available for both amd64 and arm64)
RUN apt-get update \
    && apt-get install -y --no-install-recommends chromium chromium-driver fonts-liberation \
    && rm -rf /var/lib/apt/lists/*
ENV CHROME_BIN=/usr/bin/chromium \
    CHROMEDRIVER_PATH=/usr/bin/chromedriver

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD gunicorn main:app --bind 0.0.0.0:${PORT:-10000} --timeout 600 --workers 1 --worker-class gthread --threads 4

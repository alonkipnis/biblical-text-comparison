# Minimal Dockerfile for the restored Streamlit app

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps: add build tools only if needed; wheels should satisfy scipy/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY restored_app/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy source repo (only what’s needed at runtime)
COPY restored_app /app/restored_app
COPY bib-scripts/data/01_raw /app/bib-scripts/data/01_raw

EXPOSE 8501

CMD ["streamlit", "run", "restored_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]


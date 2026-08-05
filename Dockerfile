# Build Stage for Dependencies
FROM python:3.11-slim as builder

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements-related files first for Docker caching
COPY pyproject.toml .

# Install dependencies into the system python
RUN pip install --no-cache-dir .

# Final Stage
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy installed python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user
RUN useradd -m beyondml && \
    mkdir -p /app/workspace /app/data && \
    chown -R beyondml:beyondml /app

USER beyondml

# Copy project source code (owned by beyondml user)
COPY --chown=beyondml:beyondml . /app/

# The command that will be executed when the container starts
CMD ["beyondml", "run"]

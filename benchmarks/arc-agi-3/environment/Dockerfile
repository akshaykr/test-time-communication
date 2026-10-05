FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ARCCTL_HOST=arc-server \
    ARCCTL_PORT=8765 \
    ARCCTL_DEFAULT_GAME=ls20

RUN useradd --create-home --shell /bin/bash arcagent

WORKDIR /app

COPY arcctl.py /usr/local/bin/arcctl
RUN chmod +x /usr/local/bin/arcctl

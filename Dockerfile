# ===== STAGE 1: Builder (instala dependências) =====
FROM python:3.12-slim as builder

WORKDIR /app

# Instalar apenas ferramentas necessárias para compilação
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar em um virtualenv
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt


# ===== STAGE 2: Runtime (imagem final) =====
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/src \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Instalar apenas curl (necessário para healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar venv do builder (sem ferramentas de build!)
COPY --from=builder /opt/venv /opt/venv

# Código-fonte
COPY src/ src/
COPY api/ api/
COPY models/ models/
COPY data/resumo_categorias.json data/resumo_categorias.json

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "api/api.py", "--server.port=8501", "--server.address=0.0.0.0"]
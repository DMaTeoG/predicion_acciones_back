# Dockerfile para Backend Django + gRPC Engine
FROM python:3.11-slim

# Evitar escritura de bytecode y habilitar buffer stdout directo para logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Instalar dependencias del sistema requeridas para compilar paquetes C/C++ y PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY . /app/

# Exponer puertos: 8000 (Django API REST) y 50051 (gRPC Microservice)
EXPOSE 8000 50051

# Comando por defecto (aplicar migraciones e iniciar servidor Gunicorn/Django)
CMD ["sh", "-c", "python manage.py prepare_db && gunicorn config.wsgi:application --config gunicorn.conf.py --bind 0.0.0.0:${PORT:-8000}"]

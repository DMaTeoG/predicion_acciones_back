import os

# Optimización para Render Free Tier (Límite estricto de 512 MB de RAM)
# Un solo proceso worker con múltiples hilos (gthread) comparte la memoria de Python,
# evitando que Django + NumPy + Pandas dupliquen su espacio de memoria.
workers = int(os.getenv("WEB_CONCURRENCY", 1))
threads = int(os.getenv("GUNICORN_THREADS", 4))
worker_class = "gthread"

# Timeout para solicitudes largas y streaming SSE
timeout = int(os.getenv("GUNICORN_TIMEOUT", 120))
keepalive = 5

# Reciclaje automático del worker para prevenir fugas de memoria
max_requests = 1000
max_requests_jitter = 50

# Logs directos a stdout/stderr
accesslog = "-"
errorlog = "-"
loglevel = "info"

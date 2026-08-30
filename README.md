# predicion_acciones_back

Servicio Backend & Motor de Inteligencia Cuantitativa (NLP & Machine Learning) para la predicción probabilística de activos financieros en tiempo real.

## Arquitectura y Componentes
- **Framework Web**: Django 5.x & Django REST Framework (DRF).
- **Autenticación**: JWT (JSON Web Tokens) con refresh y access tokens (`SimpleJWT`).
- **Motor NLP**: Análisis de sentimiento financiero VADER & TextBlob con indexación multicanal de noticias RSS en vivo.
- **Microservicio gRPC**: Comunicación de baja latencia con el motor probabilístico de inferencia (`apps.ml_engine`).
- **Streaming en Vivo**: Server-Sent Events (SSE) en `/api/v1/stream/quote/` con microestructura de mercado en tiempo real.
- **Exportación Multifactorial**: Generación de reportes ejecutivos descargables en CSV y PDF.

## Instalación y Ejecución Local
```bash
python -m venv venv
venv\Scripts\activate  # En Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver 8000
```

## Pruebas Automatizadas
```bash
python manage.py test tests.backend -v 2
```

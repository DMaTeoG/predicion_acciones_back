import logging
import os
from django.core.management import call_command
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

application = get_wsgi_application()

logger = logging.getLogger(__name__)

# Garantizar que las migraciones y datos semilla estén aplicados en producción (Render/Docker)
# incluso si el Build Command del proveedor no ejecutó build.sh.
if os.getenv('AUTO_MIGRATE_ON_STARTUP', 'True') == 'True':
    try:
        call_command('prepare_db')
    except Exception as exc:
        logger.warning("Auto-migrate en arranque WSGI omitido o con advertencia: %s", exc)


from django.contrib.auth.models import Group, Permission, User
from django.contrib.contenttypes.models import ContentType
from django.contrib.sessions.models import Session
from django.core.management import BaseCommand, call_command
from django.db import connection

from apps.core.models import (
    Asset,
    MarketData,
    ModelMetrics,
    NewsArticle,
    NewsSentiment,
    Prediction,
    UserScan,
)


class Command(BaseCommand):
    help = (
        "Prepara y repara el esquema de base de datos en entornos compartidos (ej. Render PostgreSQL) "
        "donde auth_user puede no existir si otro proyecto usó un AUTH_USER_MODEL personalizado."
    )

    def handle(self, *args, **options):
        existing_tables = set(connection.introspection.table_names())

        # 1. Asegurar tablas base de Django (especialmente auth_user si la BD fue inicializada con otro AUTH_USER_MODEL)
        base_models = [ContentType, Permission, Group, User, Session]
        for model in base_models:
            table_name = model._meta.db_table
            if table_name not in existing_tables:
                self.stdout.write(self.style.WARNING(f"Creando tabla base faltante: {table_name}..."))
                try:
                    with connection.schema_editor() as schema_editor:
                        schema_editor.create_model(model)
                    existing_tables.add(table_name)
                except Exception as exc:
                    self.stdout.write(self.style.WARNING(f"Aviso al crear {table_name}: {exc}"))

        # 2. Ejecutar migraciones estándar de Django
        try:
            call_command("migrate", interactive=False, verbosity=1)
        except Exception as exc:
            self.stdout.write(self.style.WARNING(f"Aviso durante migrate estándar: {exc}"))

        # 3. Verificar que todas las tablas de apps.core existan
        existing_tables = set(connection.introspection.table_names())
        core_models = [
            Asset,
            NewsArticle,
            NewsSentiment,
            MarketData,
            Prediction,
            ModelMetrics,
            UserScan,
        ]
        for model in core_models:
            table_name = model._meta.db_table
            if table_name not in existing_tables:
                self.stdout.write(self.style.WARNING(f"Creando tabla core faltante: {table_name}..."))
                try:
                    with connection.schema_editor() as schema_editor:
                        schema_editor.create_model(model)
                    existing_tables.add(table_name)
                except Exception as exc:
                    self.stdout.write(self.style.WARNING(f"Aviso al crear {table_name}: {exc}"))

        # 4. Sembrar activos iniciales
        try:
            call_command("seed_data")
        except Exception as exc:
            self.stdout.write(self.style.WARNING(f"Aviso durante seed_data: {exc}"))

        self.stdout.write(self.style.SUCCESS("Base de datos preparada y verificada correctamente."))

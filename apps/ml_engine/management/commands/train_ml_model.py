from django.core.management.base import BaseCommand, CommandError
from apps.ml_engine.trainer import MLTrainer


class Command(BaseCommand):
    help = 'Entrena y evalúa el modelo de Machine Learning con validación temporal por fecha'

    def add_arguments(self, parser):
        parser.add_argument('--model-version', type=str, default='v1.0.0', help='Versión del modelo')

    def handle(self, *args, **options):
        version = options['model_version']
        self.stdout.write(self.style.NOTICE(f'Iniciando entrenamiento del modelo {version}...'))

        result = MLTrainer(version=version).train_and_evaluate()

        if result.get('status') != 'success':
            raise CommandError(f"Error en entrenamiento: {result.get('message')}")

        self.stdout.write(self.style.SUCCESS(
            f"Entrenamiento exitoso | Modelo {result['model_version']} | "
            f"Accuracy (fuera de muestra): {result['accuracy']:.2%} | "
            f"F1-Score: {result['f1_score']:.2f} | "
            f"Muestras: {result['samples_trained']} (validación: {result['validation_samples']}) | "
            f"Fechas: {result['unique_dates']} | Folds: {result['n_splits']}"
        ))

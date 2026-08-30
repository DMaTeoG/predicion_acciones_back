from django.core.management.base import BaseCommand, CommandError
from apps.core.models import Asset
from apps.ml_engine.predictor import MLPredictor, PredictionUnavailable


class Command(BaseCommand):
    help = 'Genera y almacena predicciones probabilísticas para todos los activos activos'

    def add_arguments(self, parser):
        parser.add_argument('--model-version', type=str, default='v1.0.0', help='Versión del modelo a usar')

    def handle(self, *args, **options):
        version = options['model_version']
        predictor = MLPredictor(version=version)

        try:
            predictor.load_model()
        except PredictionUnavailable as exc:
            raise CommandError(str(exc)) from exc

        assets = Asset.objects.filter(is_active=True)
        self.stdout.write(self.style.NOTICE(
            f'Generando predicciones con modelo {version} para {assets.count()} activos...'
        ))

        created, failures = 0, []
        for asset in assets:
            try:
                pred = predictor.predict_asset(asset)
            except PredictionUnavailable as exc:
                failures.append(f'{asset.ticker}: {exc}')
                self.stderr.write(self.style.ERROR(f'[{asset.ticker}] {exc}'))
                continue
            created += 1
            self.stdout.write(self.style.SUCCESS(
                f"[{asset.ticker}] Predicción: {pred.predicted_class} | "
                f"Objetivo: {pred.target_date:%Y-%m-%d} | "
                f"Prob Sube: {pred.prob_sube:.1%} | Prob Estable: {pred.prob_estable:.1%} | "
                f"Prob Baja: {pred.prob_baja:.1%}"
            ))

        self.stdout.write(f'Predicciones creadas: {created} | Fallidas: {len(failures)}')
        if failures:
            raise CommandError('No se pudo predecir para: ' + '; '.join(failures))

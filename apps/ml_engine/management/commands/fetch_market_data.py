from django.core.management.base import BaseCommand, CommandError
from apps.core.models import Asset
from apps.ml_engine.market_fetcher import MarketDataUnavailable, MarketFetcher

class Command(BaseCommand):
    help = 'Sincroniza precios e indicadores técnicos de mercado para los activos activos'

    def handle(self, *args, **options):
        fetcher = MarketFetcher()
        assets = Asset.objects.filter(is_active=True)
        total_records = 0
        errors = []

        self.stdout.write(self.style.NOTICE(f'Iniciando sincronización para {assets.count()} activos...'))

        for asset in assets:
            try:
                count = fetcher.sync_asset_market_data(asset)
            except MarketDataUnavailable as exc:
                message = f'[{asset.ticker}] Error de sincronización: {exc}'
                errors.append(message)
                self.stderr.write(self.style.ERROR(message))
                continue

            total_records += count
            self.stdout.write(self.style.SUCCESS(f'[{asset.ticker}] Registros procesados: {count}'))

        if errors:
            raise CommandError(
                f'Sincronización incompleta: {len(errors)} activo(s) fallaron; '
                f'{total_records} registro(s) procesados. '
                + ' | '.join(errors)
            )

        self.stdout.write(self.style.SUCCESS(f'Sincronización completa. Total de registros procesados: {total_records}'))

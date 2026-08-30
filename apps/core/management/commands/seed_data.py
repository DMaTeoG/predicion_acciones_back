from django.core.management.base import BaseCommand
from apps.core.models import Asset

class Command(BaseCommand):
    help = 'Crea los activos financieros iniciales (Acciones y Criptomonedas)'

    def handle(self, *args, **options):
        initial_assets = [
            {'ticker': 'AAPL', 'name': 'Apple Inc.', 'asset_type': Asset.AssetType.STOCK},
            {'ticker': 'TSLA', 'name': 'Tesla, Inc.', 'asset_type': Asset.AssetType.STOCK},
            {'ticker': 'BTC-USD', 'name': 'Bitcoin USD', 'asset_type': Asset.AssetType.CRYPTO},
        ]

        active_tickers = [data['ticker'] for data in initial_assets]
        deactivated_count = Asset.objects.filter(is_active=True).exclude(
            ticker__in=active_tickers
        ).update(is_active=False)

        created_count = 0
        for data in initial_assets:
            asset, created = Asset.objects.update_or_create(
                ticker=data['ticker'],
                defaults={
                    'name': data['name'],
                    'asset_type': data['asset_type'],
                    'is_active': True,
                },
            )
            if created:
                created_count += 1
                self.stdout.write(self.style.SUCCESS(f'Activo creado: {asset.ticker} ({asset.name})'))
            else:
                self.stdout.write(f'Activo actualizado: {asset.ticker}')

        self.stdout.write(self.style.SUCCESS(
            f'Sembrado finalizado. Activos nuevos creados: {created_count}; '
            f'activos previos desactivados: {deactivated_count}'
        ))

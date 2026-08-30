from django.core.management.base import BaseCommand, CommandError
from apps.core.models import Asset, NewsArticle
from apps.scraper.pipelines import DjangoNewsPipeline
from apps.scraper.rss_client import GoogleNewsRssClient, RSSFeedError

class Command(BaseCommand):
    help = 'Ingiere noticias RSS financieras y calcula sentimiento NLP'

    def add_arguments(self, parser):
        parser.add_argument(
            '--ticker',
            type=str,
            help='Limita la ingesta a un activo (por ejemplo AAPL)',
        )

    def handle(self, *args, **options):
        assets = Asset.objects.filter(is_active=True).order_by('ticker')
        ticker = options.get('ticker')
        if ticker:
            assets = assets.filter(ticker__iexact=ticker)
            if not assets.exists():
                raise CommandError(f'No existe un activo activo con ticker {ticker}')
        elif not assets.exists():
            raise CommandError('No hay activos activos para ingestar noticias')

        client = GoogleNewsRssClient()
        pipeline = DjangoNewsPipeline()
        totals = {'insertadas': 0, 'duplicadas': 0, 'descartadas': 0}
        errors = []

        self.stdout.write(self.style.NOTICE('Iniciando ingesta RSS y análisis NLP...'))
        for asset in assets:
            try:
                items = client.fetch(asset)
            except RSSFeedError as exc:
                message = f'[{asset.ticker}] Error RSS: {exc}'
                errors.append(message)
                self.stderr.write(self.style.ERROR(message))
                continue

            counts = {'insertadas': 0, 'duplicadas': 0, 'descartadas': 0}
            for item in items:
                url = (item.get('url') or '').strip()
                existed_before = bool(url) and NewsArticle.objects.filter(url=url).exists()
                pipeline.process_item(item, spider=None)
                exists_after = bool(url) and NewsArticle.objects.filter(url=url).exists()

                if existed_before:
                    counts['duplicadas'] += 1
                elif exists_after:
                    counts['insertadas'] += 1
                else:
                    counts['descartadas'] += 1

            for key, value in counts.items():
                totals[key] += value
            self.stdout.write(
                f"[{asset.ticker}] insertadas: {counts['insertadas']}, "
                f"duplicadas: {counts['duplicadas']}, "
                f"descartadas: {counts['descartadas']}"
            )

        self.stdout.write(
            f"Totales RSS: insertadas: {totals['insertadas']}, "
            f"duplicadas: {totals['duplicadas']}, "
            f"descartadas: {totals['descartadas']}"
        )
        if errors:
            raise CommandError(
                f"Ingesta incompleta: {len(errors)} feed(s) fallaron. "
                + ' | '.join(errors)
            )
        self.stdout.write(self.style.SUCCESS('Ingesta RSS y NLP completada.'))

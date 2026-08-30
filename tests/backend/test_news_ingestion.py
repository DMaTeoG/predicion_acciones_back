from datetime import datetime, timezone as dt_timezone
from importlib import import_module, util
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock, patch
from requests.exceptions import ConnectionError
from urllib.parse import parse_qs, urlparse
from django.core.management import call_command, CommandError
from django.test import SimpleTestCase, TestCase
from django.utils import timezone as django_timezone
from apps.core.models import Asset, NewsArticle, NewsSentiment
from apps.scraper.pipelines import DjangoNewsPipeline
from apps.scraper.rss_client import RSSFeedError


RSS_XML = b'''<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <item>
    <title>Apple reports record services revenue</title>
    <link>https://example.com/aapl-news</link>
    <description>Quarterly revenue exceeded estimates.</description>
    <pubDate>Sat, 17 Oct 2026 08:00:00 GMT</pubDate>
    <source url="https://reuters.com">Reuters</source>
  </item>
</channel></rss>'''


class RSSParserTests(SimpleTestCase):
    def _parse(self, xml_bytes, ticker='AAPL'):
        module_spec = util.find_spec('apps.scraper.rss_client')
        self.assertIsNotNone(module_spec, 'apps.scraper.rss_client must exist')
        module = import_module('apps.scraper.rss_client')
        parser = getattr(module, 'parse_rss_items', None)
        self.assertIsNotNone(parser, 'rss_client.parse_rss_items must exist')
        return parser(xml_bytes, ticker)

    def test_parse_preserves_article_fields_and_published_timestamp(self):
        articles = self._parse(RSS_XML)

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]['ticker'], 'AAPL')
        self.assertEqual(articles[0]['title'], 'Apple reports record services revenue')
        self.assertEqual(articles[0]['summary'], 'Quarterly revenue exceeded estimates.')
        self.assertEqual(articles[0]['url'], 'https://example.com/aapl-news')
        self.assertEqual(articles[0]['source'], 'Reuters')
        self.assertEqual(
            articles[0]['published_at'],
            datetime(2026, 10, 17, 8, 0, tzinfo=dt_timezone.utc),
        )


class RSSClientTests(SimpleTestCase):
    def _client_and_module(self):
        module_spec = util.find_spec('apps.scraper.rss_client')
        self.assertIsNotNone(module_spec, 'apps.scraper.rss_client must exist')
        module = import_module('apps.scraper.rss_client')
        client_class = getattr(module, 'GoogleNewsRssClient', None)
        self.assertIsNotNone(client_class, 'rss_client.GoogleNewsRssClient must exist')
        return client_class(), module

    @patch('apps.scraper.rss_client.requests.get')
    def test_fetch_encodes_asset_query_and_uses_timeout(self, get_mock):
        client, _ = self._client_and_module()
        response = Mock()
        response.content = RSS_XML
        get_mock.return_value = response
        asset = SimpleNamespace(ticker='AAPL', name='Apple Inc.')

        articles = client.fetch(asset)

        get_mock.assert_called_once()
        requested_url = get_mock.call_args.args[0]
        query = parse_qs(urlparse(requested_url).query)
        self.assertIn('AAPL', query['q'][0])
        self.assertIn('Apple Inc.', query['q'][0])
        self.assertEqual(get_mock.call_args.kwargs['timeout'], 10)
        self.assertEqual(articles[0]['ticker'], 'AAPL')

    @patch('apps.scraper.rss_client.requests.get', side_effect=ConnectionError('offline'))
    def test_fetch_wraps_network_errors(self, get_mock):
        client, module = self._client_and_module()
        asset = SimpleNamespace(ticker='AAPL', name='Apple Inc.')

        with self.assertRaises(module.RSSFeedError):
            client.fetch(asset)

        get_mock.assert_called_once()


class NewsPipelineTests(TestCase):
    def setUp(self):
        self.aapl = Asset.objects.create(
            ticker='AAPL',
            name='Apple Inc.',
            asset_type=Asset.AssetType.STOCK,
        )
        self.tsla = Asset.objects.create(
            ticker='TSLA',
            name='Tesla, Inc.',
            asset_type=Asset.AssetType.STOCK,
        )
        self.pipeline = DjangoNewsPipeline()

    def test_rss_ticker_controls_assignment_and_duplicate_run_is_idempotent(self):
        item = {
            'ticker': 'TSLA',
            'title': 'Quarterly results beat market estimates',
            'summary': 'Revenue increased during the quarter.',
            'url': 'https://example.com/tsla-rss',
            'source': 'Google News RSS',
            'published_at': django_timezone.now(),
        }

        self.pipeline.process_item(item, spider=None)
        self.pipeline.process_item(item, spider=None)

        article = NewsArticle.objects.get(url=item['url'])
        self.assertEqual(article.asset, self.tsla)
        self.assertEqual(NewsArticle.objects.filter(url=item['url']).count(), 1)
        self.assertEqual(NewsSentiment.objects.filter(article=article).count(), 1)

    def test_unmatched_article_is_not_assigned_to_an_arbitrary_asset(self):
        item = {
            'title': 'Quarterly economic outlook released',
            'summary': 'A macroeconomic research note.',
            'url': 'https://example.com/unmatched-news',
            'source': 'Google News RSS',
            'published_at': django_timezone.now(),
        }

        self.pipeline.process_item(item, spider=None)

        self.assertFalse(NewsArticle.objects.filter(url=item['url']).exists())
        self.assertEqual(NewsSentiment.objects.count(), 0)


class RunScraperCommandTests(TestCase):
    def setUp(self):
        self.asset = Asset.objects.create(
            ticker='TSLA',
            name='Tesla, Inc.',
            asset_type=Asset.AssetType.STOCK,
            is_active=True,
        )
        self.item = {
            'ticker': 'TSLA',
            'title': 'Tesla reports higher deliveries',
            'summary': 'Deliveries exceeded expectations.',
            'url': 'https://example.com/tsla-deliveries',
            'source': 'Reuters',
            'published_at': django_timezone.now(),
        }

    @patch('apps.scraper.rss_client.GoogleNewsRssClient.fetch')
    def test_run_scraper_persists_rss_articles_and_sentiment(self, fetch_mock):
        fetch_mock.return_value = [self.item]
        output = StringIO()

        call_command('run_scraper', '--ticker', 'TSLA', stdout=output, stderr=output)

        article = NewsArticle.objects.get(url=self.item['url'])
        self.assertEqual(article.asset, self.asset)
        self.assertTrue(NewsSentiment.objects.filter(article=article).exists())
        fetch_mock.assert_called_once_with(self.asset)
        self.assertIn('insertadas: 1', output.getvalue())

    @patch('apps.scraper.rss_client.GoogleNewsRssClient.fetch')
    def test_rerunning_command_counts_duplicate_without_duplicate_rows(self, fetch_mock):
        fetch_mock.return_value = [self.item]
        output = StringIO()

        call_command('run_scraper', '--ticker', 'TSLA', stdout=output, stderr=output)
        call_command('run_scraper', '--ticker', 'TSLA', stdout=output, stderr=output)

        self.assertEqual(NewsArticle.objects.filter(url=self.item['url']).count(), 1)
        self.assertEqual(NewsSentiment.objects.count(), 1)
        self.assertIn('duplicadas: 1', output.getvalue())
        self.assertEqual(fetch_mock.call_count, 2)

    @patch('apps.scraper.rss_client.GoogleNewsRssClient.fetch', side_effect=RSSFeedError('offline'))
    def test_feed_failure_is_reported_as_command_error(self, fetch_mock):
        output = StringIO()

        with self.assertRaisesMessage(CommandError, 'TSLA'):
            call_command('run_scraper', '--ticker', 'TSLA', stdout=output, stderr=output)

        self.assertIn('offline', output.getvalue())
        self.assertFalse(NewsArticle.objects.exists())
        fetch_mock.assert_called_once_with(self.asset)


class IntegratedIngestionFlowTests(TestCase):
    """Tarea 21: el flujo de ingesta de mercado y noticias es idempotente y no duplica filas."""

    def setUp(self):
        self.asset = Asset.objects.create(
            ticker='AAPL',
            name='Apple Inc.',
            asset_type=Asset.AssetType.STOCK,
            is_active=True,
        )

    @patch('apps.scraper.rss_client.GoogleNewsRssClient.fetch')
    @patch('apps.ml_engine.market_fetcher.MarketFetcher.sync_asset_market_data')
    def test_combined_ingestion_can_be_safely_repeated(self, market_mock, rss_mock):
        market_mock.return_value = 10
        rss_mock.return_value = [{
            'ticker': 'AAPL',
            'title': 'Apple announces breakthrough tech',
            'summary': 'Major developments announced.',
            'url': 'https://example.com/aapl-tech',
            'source': 'Reuters',
            'published_at': django_timezone.now(),
        }]

        # Primera ejecución
        call_command('fetch_market_data', stdout=StringIO(), stderr=StringIO())
        call_command('run_scraper', '--ticker', 'AAPL', stdout=StringIO(), stderr=StringIO())

        self.assertEqual(NewsArticle.objects.filter(asset=self.asset).count(), 1)
        self.assertEqual(NewsSentiment.objects.count(), 1)

        # Segunda ejecución (reintento / ejecución diaria)
        call_command('fetch_market_data', stdout=StringIO(), stderr=StringIO())
        call_command('run_scraper', '--ticker', 'AAPL', stdout=StringIO(), stderr=StringIO())

        self.assertEqual(NewsArticle.objects.filter(asset=self.asset).count(), 1)
        self.assertEqual(NewsSentiment.objects.count(), 1)


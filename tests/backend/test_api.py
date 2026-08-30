from io import StringIO
from unittest.mock import patch
from django.conf import settings
from django.contrib.auth.models import User
from django.core.management import call_command
from django.test import SimpleTestCase, TestCase
from rest_framework.test import APITestCase
from rest_framework import status
from django.utils import timezone
from apps.core.models import Asset, NewsArticle, NewsSentiment, MarketData, Prediction, ModelMetrics, UserScan

class APIIntegrationTests(APITestCase):

    def setUp(self):
        self.asset = Asset.objects.create(
            ticker='AAPL',
            name='Apple Inc.',
            asset_type=Asset.AssetType.STOCK
        )

        now = timezone.now()
        self.article = NewsArticle.objects.create(
            asset=self.asset,
            title='Prueba Noticia API',
            summary='Resumen API',
            source='Reuters',
            url='https://example.com/api-news-1',
            published_at=now
        )
        self.sentiment = NewsSentiment.objects.create(
            article=self.article,
            sentiment_label=NewsSentiment.SentimentLabel.POSITIVE,
            sentiment_score=0.8,
            impact_score=0.9
        )
        self.market_data = MarketData.objects.create(
            asset=self.asset,
            timestamp=now,
            open_price=150.0,
            high_price=152.0,
            low_price=149.0,
            close_price=151.0,
            volume=1000000.0,
            rsi_14=55.0,
            macd=0.5,
            macd_signal=0.4,
            sma_20=150.0,
            sma_50=145.0,
            volatility_20=0.015
        )
        self.prediction = Prediction.objects.create(
            asset=self.asset,
            target_date=now,
            predicted_class=Prediction.PredictedClass.SUBE,
            prob_sube=0.70,
            prob_estable=0.20,
            prob_baja=0.10,
            model_version='v1.0.0'
        )
        self.metrics = ModelMetrics.objects.create(
            model_version='v1.0.0',
            accuracy=0.85,
            f1_score=0.82
        )

    def test_get_assets(self):
        response = self.client.get('/api/v1/assets/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertGreaterEqual(len(response.data['results']), 1)

    def test_get_news_by_asset(self):
        response = self.client.get('/api/v1/news/?asset=AAPL')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertGreaterEqual(len(response.data['results']), 1)
        self.assertEqual(response.data['results'][0]['asset_ticker'], 'AAPL')

    def test_get_market_data_by_asset(self):
        response = self.client.get('/api/v1/market-data/?asset=AAPL')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertGreaterEqual(len(response.data['results']), 1)

    def test_get_latest_predictions(self):
        response = self.client.get('/api/v1/predictions/latest/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertGreaterEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['asset_ticker'], 'AAPL')
        self.assertEqual(response.data[0]['predicted_class'], 'SUBE')

    def test_get_metrics(self):
        response = self.client.get('/api/v1/metrics/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertGreaterEqual(len(response.data['results']), 1)

    def test_prediction_history_filters_by_asset(self):
        other = Asset.objects.create(ticker='TSLA', name='Tesla', asset_type=Asset.AssetType.STOCK)
        Prediction.objects.create(
            asset=other, target_date=timezone.now(), predicted_class='BAJA',
            prob_sube=0.1, prob_estable=0.2, prob_baja=0.7, model_version='v1.0.0',
        )
        response = self.client.get('/api/v1/predictions/history/?asset=AAPL')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        tickers = {row['asset_ticker'] for row in response.data['results']}
        self.assertSetEqual(tickers, {'AAPL'})

    def test_api_is_read_only(self):
        response = self.client.post('/api/v1/assets/', {'ticker': 'X', 'name': 'X'}, format='json')
        self.assertEqual(response.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)
        response = self.client.delete(f'/api/v1/predictions/{self.prediction.pk}/')
        self.assertEqual(response.status_code, status.HTTP_405_METHOD_NOT_ALLOWED)


class CorsSettingsTests(SimpleTestCase):
    def test_cors_is_limited_to_local_frontend(self):
        self.assertFalse(getattr(settings, 'CORS_ALLOW_ALL_ORIGINS', False))
        self.assertIn('http://localhost:3001', settings.CORS_ALLOWED_ORIGINS)
        self.assertIn('http://127.0.0.1:3001', settings.CORS_ALLOWED_ORIGINS)


class SeedDataCommandTests(TestCase):
    def test_seed_keeps_only_mvp_assets_active_without_deleting_old_assets(self):
        previous_asset = Asset.objects.create(
            ticker='MSFT',
            name='Microsoft Corporation',
            asset_type=Asset.AssetType.STOCK,
            is_active=True,
        )

        call_command('seed_data', stdout=StringIO())
        call_command('seed_data', stdout=StringIO())

        active_tickers = set(
            Asset.objects.filter(is_active=True).values_list('ticker', flat=True)
        )
        self.assertSetEqual(active_tickers, {'AAPL', 'TSLA', 'BTC-USD'})
        previous_asset.refresh_from_db()
        self.assertFalse(previous_asset.is_active)
        self.assertTrue(Asset.objects.filter(pk=previous_asset.pk).exists())
        self.assertEqual(Asset.objects.count(), 4)


class AuthAndScanningApiTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='trader1', email='trader1@example.com', password='secretpassword123'
        )

    def test_register_creates_user_and_returns_tokens(self):
        payload = {
            'username': 'newtrader',
            'email': 'newtrader@example.com',
            'password': 'password12345',
        }
        response = self.client.post('/api/v1/auth/register/', payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('tokens', response.data)
        self.assertIn('access', response.data['tokens'])
        self.assertTrue(User.objects.filter(username='newtrader').exists())

    def test_login_returns_jwt_tokens(self):
        payload = {'username': 'trader1', 'password': 'secretpassword123'}
        response = self.client.post('/api/v1/auth/login/', payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('tokens', response.data)
        self.assertEqual(response.data['user']['username'], 'trader1')

    def test_login_with_invalid_credentials_returns_401(self):
        payload = {'username': 'trader1', 'password': 'wrongpassword'}
        response = self.client.post('/api/v1/auth/login/', payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_me_requires_authentication(self):
        response = self.client.get('/api/v1/auth/me/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

        self.client.force_authenticate(user=self.user)
        auth_response = self.client.get('/api/v1/auth/me/')
        self.assertEqual(auth_response.status_code, status.HTTP_200_OK)
        self.assertEqual(auth_response.data['username'], 'trader1')

    @patch('apps.ml_engine.live_scanner.LiveAssetScanner.scan')
    def test_scan_endpoint_calls_scanner_and_persists_scan(self, mock_scan):
        mock_scan.return_value = {
            'id': 1,
            'ticker': 'NVDA',
            'name': 'NVIDIA Corporation',
            'current_price': 130.5,
            'prediction': {'predicted_class': 'SUBE', 'confidence_pct': 72.5},
            'advice': 'Convergencia alcista recomendada.',
            'top_news': [],
        }
        response = self.client.post('/api/v1/scan/', {'query': 'NVDA'}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['ticker'], 'NVDA')
        mock_scan.assert_called_once()

    @patch('apps.ml_engine.live_scanner.LiveAssetScanner.scan')
    def test_scan_endpoint_with_keywords(self, mock_scan):
        mock_scan.return_value = {
            'id': 2,
            'ticker': 'TSLA',
            'name': 'Tesla, Inc.',
            'keywords': 'robotaxi',
            'query_type': 'ticker_with_keywords',
            'current_price': 220.0,
            'prediction': {'predicted_class': 'SUBE', 'confidence_pct': 68.0},
            'advice': 'Impulso favorable.',
            'top_news': [],
        }
        response = self.client.post('/api/v1/scan/', {'query': 'TSLA', 'keywords': 'robotaxi'}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['keywords'], 'robotaxi')
        mock_scan.assert_called_once_with(query='TSLA', user=None, keywords='robotaxi')

    @patch('apps.ml_engine.live_scanner.LiveAssetScanner.scan')
    def test_scan_endpoint_with_theme(self, mock_scan):
        mock_scan.return_value = {
            'id': 3,
            'ticker': 'NVDA',
            'name': 'Nvidia AI Leader',
            'theme_label': 'Inteligencia Artificial',
            'query_type': 'theme',
            'current_price': 135.0,
            'prediction': {'predicted_class': 'SUBE', 'confidence_pct': 80.0},
            'advice': 'Liderazgo en inteligencia artificial.',
            'top_news': [],
        }
        response = self.client.get('/api/v1/scan/?query=Inteligencia+Artificial')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['theme_label'], 'Inteligencia Artificial')

    def test_live_quote_stream_endpoint(self):
        response = self.client.get('/api/v1/stream/quote/?ticker=AAPL')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response['Content-Type'], 'text/event-stream')
        # Check first streaming chunk
        iterator = response.streaming_content
        first_chunk = next(iter(iterator))
        self.assertIn(b'event: connected', first_chunk)

    def test_export_csv_endpoint(self):
        # Create a sample UserScan in db
        scan = UserScan.objects.create(
            ticker='AAPL',
            asset_name='Apple Inc.',
            current_price=180.5,
            price_change_pct=1.2,
            currency='USD',
            predicted_class='SUBE',
            prob_sube=0.7,
            prob_estable=0.2,
            prob_baja=0.1,
            advice='Fuerte tendencia alcista.',
            top_news=[{'title': 'Apple Q4 Record', 'source': 'Reuters', 'sentiment_label': 'Positivo', 'sentiment_score': 0.8}],
            market_snapshot={'rsi_14': 60.5, 'macd': 1.2, 'sma_20': 178.0},
        )
        response = self.client.get(f'/api/v1/export/csv/?scan_id={scan.id}')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('text/csv', response['Content-Type'])
        self.assertIn('attachment;', response['Content-Disposition'])
        content = response.content.decode('utf-8')
        self.assertIn('AAPL', content)
        self.assertIn('Apple Inc.', content)
        self.assertIn('SUBE', content)

    def test_export_executive_report_endpoint(self):
        scan = UserScan.objects.create(
            ticker='NVDA',
            asset_name='NVIDIA Corp.',
            current_price=210.0,
            price_change_pct=3.5,
            currency='USD',
            predicted_class='SUBE',
            prob_sube=0.85,
            prob_estable=0.10,
            prob_baja=0.05,
            advice='Impulso por IA y centros de datos.',
            top_news=[{'title': 'NVIDIA New GPU', 'source': 'Bloomberg', 'sentiment_label': 'Positivo', 'sentiment_score': 0.9}],
            market_snapshot={'rsi_14': 68.0, 'macd': 2.5, 'sma_20': 200.0},
        )
        response = self.client.get(f'/api/v1/export/executive/?scan_id={scan.id}')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('text/html', response['Content-Type'])
        content = response.content.decode('utf-8')
        self.assertIn('NVIDIA Corp.', content)
        self.assertIn('NVDA', content)
        self.assertIn('SUBE', content)
        self.assertIn('NVIDIA New GPU', content)

    def test_export_csv_fallback_by_ticker(self):
        response = self.client.get('/api/v1/export/csv/?ticker=AAPL')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('text/csv', response['Content-Type'])



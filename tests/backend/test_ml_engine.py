import math
import tempfile
from io import StringIO
from unittest.mock import patch
import pandas as pd
from django.core.management import call_command, CommandError
from django.test import SimpleTestCase, TestCase, override_settings
from django.utils import timezone
from datetime import datetime, timedelta, timezone as dt_timezone
from apps.core.models import Asset, MarketData, NewsArticle, NewsSentiment, Prediction, ModelMetrics
from apps.ml_engine.trainer import MLTrainer
from apps.ml_engine.predictor import MLPredictor, PredictionUnavailable, next_target_date
from apps.ml_engine.market_fetcher import MarketFetcher, MarketDataUnavailable

class TempArtifactsMixin:
    """Redirige ML_ARTIFACTS_DIR a un directorio temporal para no ensuciar el repo."""

    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        override = self.settings(ML_ARTIFACTS_DIR=tmp.name)
        override.enable()
        self.addCleanup(override.disable)


class MLEngineTests(TempArtifactsMixin, TestCase):

    def setUp(self):
        super().setUp()
        self.asset = Asset.objects.create(
            ticker='AAPL',
            name='Apple Inc.',
            asset_type=Asset.AssetType.STOCK
        )

        now = timezone.now()
        # Create 30 rows of market data to allow TimeSeriesSplit
        for i in range(30):
            MarketData.objects.create(
                asset=self.asset,
                timestamp=now - timedelta(days=30 - i),
                open_price=150.0 + (i * 0.5),
                high_price=152.0 + (i * 0.5),
                low_price=149.0 + (i * 0.5),
                close_price=151.0 + (i * 0.5),
                volume=1000000.0,
                rsi_14=50.0 + (i % 10),
                macd=0.1 * i,
                macd_signal=0.08 * i,
                sma_20=150.0,
                sma_50=145.0,
                volatility_20=0.015
            )

        # Create news article & sentiment
        article = NewsArticle.objects.create(
            asset=self.asset,
            title='Apple anuncia sólidas métricas operativas',
            summary='Positivo para inversores.',
            source='Reuters',
            url='https://example.com/aapl-test-ml',
            published_at=now - timedelta(days=15)
        )
        NewsSentiment.objects.create(
            article=article,
            sentiment_label=NewsSentiment.SentimentLabel.POSITIVE,
            sentiment_score=0.75,
            impact_score=0.85
        )

    def test_ml_trainer_execution(self):
        trainer = MLTrainer(version='test_v1.0')
        result = trainer.train_and_evaluate()
        self.assertEqual(result['status'], 'success')
        self.assertGreaterEqual(result['accuracy'], 0.0)
        self.assertGreaterEqual(result['f1_score'], 0.0)
        
        # Verify ModelMetrics created in DB
        metrics = ModelMetrics.objects.filter(model_version='test_v1.0').first()
        self.assertIsNotNone(metrics)

    def test_ml_predictor_execution(self):
        # Train first
        trainer = MLTrainer(version='test_v1.0')
        trainer.train_and_evaluate()

        # Predict
        predictor = MLPredictor(version='test_v1.0')
        pred = predictor.predict_asset(self.asset)

        self.assertIsNotNone(pred)
        self.assertIn(pred.predicted_class, ['SUBE', 'ESTABLE', 'BAJA'])
        probs = (pred.prob_sube, pred.prob_estable, pred.prob_baja)
        self.assertTrue(all(math.isfinite(p) for p in probs))
        self.assertAlmostEqual(sum(probs), 1.0, places=4)

    def test_metrics_are_computed_out_of_fold(self):
        result = MLTrainer(version='test_oof').train_and_evaluate()
        self.assertEqual(result['status'], 'success')
        self.assertLessEqual(result['accuracy'], 1.0)
        self.assertLessEqual(result['f1_score'], 1.0)
        cm_total = sum(sum(row) for row in result['confusion_matrix'])
        # La matriz solo cuenta muestras de validación, nunca las de entrenamiento.
        self.assertEqual(cm_total, result['validation_samples'])
        self.assertLess(result['validation_samples'], result['samples_trained'])
        metrics = ModelMetrics.objects.get(model_version='test_oof')
        self.assertEqual(metrics.confusion_matrix, result['confusion_matrix'])

    def test_insufficient_dates_return_error_without_artifacts(self):
        keep = list(MarketData.objects.filter(asset=self.asset).order_by('-timestamp').values_list('id', flat=True)[:5])
        MarketData.objects.exclude(id__in=keep).delete()
        result = MLTrainer(version='test_tiny').train_and_evaluate()
        self.assertEqual(result['status'], 'error')
        self.assertFalse(ModelMetrics.objects.filter(model_version='test_tiny').exists())


class TemporalFoldTests(SimpleTestCase):
    """Tarea 13: ningún día puede quedar a ambos lados del corte train/validación."""

    def test_shared_dates_never_split_across_train_and_validation(self):
        base = datetime(2026, 1, 1, tzinfo=dt_timezone.utc)
        stamps = []
        for day in range(20):
            # Acción (05:00 UTC) y cripto (00:00 UTC) comparten fecha de calendario.
            stamps.append(base + timedelta(days=day, hours=5))
            stamps.append(base + timedelta(days=day))
        timestamps = pd.Series(stamps).sort_values().reset_index(drop=True)
        dates = pd.to_datetime(timestamps, utc=True).dt.date

        folds = list(MLTrainer.temporal_folds(timestamps, n_splits=3))
        self.assertEqual(len(folds), 3)
        for train_idx, val_idx in folds:
            train_dates = set(dates.iloc[train_idx])
            val_dates = set(dates.iloc[val_idx])
            self.assertFalse(train_dates & val_dates)
            self.assertLess(max(train_dates), min(val_dates))


class PredictorRequirementTests(TempArtifactsMixin, TestCase):
    """Tarea 15: sin modelo o sin datos recientes no se persiste ninguna predicción."""

    def setUp(self):
        super().setUp()
        self.asset = Asset.objects.create(
            ticker='TSLA', name='Tesla', asset_type=Asset.AssetType.STOCK,
        )

    def test_missing_model_raises_and_persists_nothing(self):
        MarketData.objects.create(
            asset=self.asset, timestamp=timezone.now() - timedelta(days=1),
            open_price=1, high_price=1, low_price=1, close_price=1, volume=1,
        )
        with self.assertRaises(PredictionUnavailable):
            MLPredictor(version='does-not-exist').predict_asset(self.asset)
        self.assertFalse(Prediction.objects.exists())

    def test_missing_market_data_raises_and_persists_nothing(self):
        predictor = MLPredictor(version='any')
        predictor.artifact = {'model': object(), 'features': MLTrainer.FEATURE_COLS}
        with self.assertRaises(PredictionUnavailable):
            predictor.predict_asset(self.asset)
        self.assertFalse(Prediction.objects.exists())

    @override_settings(ML_MAX_DATA_AGE_DAYS=7)
    def test_stale_market_data_raises_and_persists_nothing(self):
        MarketData.objects.create(
            asset=self.asset, timestamp=timezone.now() - timedelta(days=30),
            open_price=1, high_price=1, low_price=1, close_price=1, volume=1,
        )
        predictor = MLPredictor(version='any')
        predictor.artifact = {'model': object(), 'features': MLTrainer.FEATURE_COLS}
        with self.assertRaises(PredictionUnavailable):
            predictor.predict_asset(self.asset)
        self.assertFalse(Prediction.objects.exists())

    def test_generate_predictions_command_fails_without_model(self):
        Asset.objects.filter(pk=self.asset.pk).update(is_active=True)
        with self.assertRaises(CommandError):
            call_command('generate_predictions', '--model-version', 'does-not-exist',
                         stdout=StringIO(), stderr=StringIO())
        self.assertFalse(Prediction.objects.exists())


class TargetDateTests(SimpleTestCase):
    """Tarea 16: horizonte = siguiente barra diaria según el tipo de activo."""

    def test_crypto_targets_next_calendar_day(self):
        friday = datetime(2026, 1, 2, tzinfo=dt_timezone.utc)
        self.assertEqual(next_target_date(Asset.AssetType.CRYPTO, friday), friday + timedelta(days=1))

    def test_stock_on_friday_targets_monday(self):
        friday = datetime(2026, 1, 2, 5, tzinfo=dt_timezone.utc)
        self.assertEqual(next_target_date(Asset.AssetType.STOCK, friday), friday + timedelta(days=3))

    def test_stock_midweek_targets_next_day(self):
        tuesday = datetime(2026, 1, 6, 5, tzinfo=dt_timezone.utc)
        self.assertEqual(next_target_date(Asset.AssetType.STOCK, tuesday), tuesday + timedelta(days=1))


class MarketFetcherFailureTests(TestCase):
    def setUp(self):
        self.asset = Asset.objects.create(
            ticker='AAPL',
            name='Apple Inc.',
            asset_type=Asset.AssetType.STOCK,
        )

    @patch('apps.ml_engine.market_fetcher.yf.Ticker')
    def test_provider_exception_is_not_replaced_with_synthetic_bars(self, ticker_mock):
        ticker_mock.return_value.history.side_effect = RuntimeError('provider unavailable')

        with self.assertRaises(RuntimeError):
            MarketFetcher().sync_asset_market_data(self.asset)

        self.assertFalse(MarketData.objects.filter(asset=self.asset).exists())

    @patch('apps.ml_engine.market_fetcher.yf.Ticker')
    def test_empty_provider_response_does_not_create_synthetic_bars(self, ticker_mock):
        ticker_mock.return_value.history.return_value = pd.DataFrame()

        with self.assertRaises(RuntimeError):
            MarketFetcher().sync_asset_market_data(self.asset)

        self.assertFalse(MarketData.objects.filter(asset=self.asset).exists())


class MarketFetcherValidationTests(TestCase):
    def setUp(self):
        self.asset = Asset.objects.create(
            ticker='AAPL',
            name='Apple Inc.',
            asset_type=Asset.AssetType.STOCK,
        )

    @patch('apps.ml_engine.market_fetcher.yf.Ticker')
    def test_missing_volume_column_is_rejected_without_persisting_bars(self, ticker_mock):
        frame = pd.DataFrame(
            {
                'Open': [10.0],
                'High': [11.0],
                'Low': [9.0],
                'Close': [10.5],
            },
            index=pd.DatetimeIndex(['2026-01-02'], name='Date'),
        )
        ticker_mock.return_value.history.return_value = frame

        with self.assertRaises(RuntimeError):
            MarketFetcher().sync_asset_market_data(self.asset)

        self.assertFalse(MarketData.objects.filter(asset=self.asset).exists())

    @patch('apps.ml_engine.market_fetcher.yf.Ticker')
    def test_inconsistent_ohlc_is_rejected_without_persisting_bars(self, ticker_mock):
        frame = pd.DataFrame(
            {
                'Open': [10.0],
                'High': [9.0],
                'Low': [8.0],
                'Close': [10.5],
                'Volume': [100.0],
            },
            index=pd.DatetimeIndex(['2026-01-02'], name='Datetime'),
        )
        ticker_mock.return_value.history.return_value = frame

        with self.assertRaises(RuntimeError):
            MarketFetcher().sync_asset_market_data(self.asset)

        self.assertFalse(MarketData.objects.filter(asset=self.asset).exists())

    @patch('apps.ml_engine.market_fetcher.yf.Ticker')
    def test_datetime_column_is_utc_sorted_and_deduplicated(self, ticker_mock):
        frame = pd.DataFrame(
            {
                'Open': [12.0, 10.0, 11.0],
                'High': [13.0, 11.0, 12.0],
                'Low': [11.0, 9.0, 10.0],
                'Close': [12.5, 10.5, 11.5],
                'Volume': [120.0, 100.0, 110.0],
            },
            index=pd.DatetimeIndex(
                ['2026-01-02', '2026-01-01', '2026-01-01'], name='Datetime'
            ),
        )
        ticker_mock.return_value.history.return_value = frame

        processed = MarketFetcher().sync_asset_market_data(self.asset)

        self.assertEqual(processed, 2)
        rows = list(MarketData.objects.filter(asset=self.asset).order_by('timestamp'))
        self.assertEqual([row.close_price for row in rows], [11.5, 12.5])
        self.assertTrue(all(row.timestamp.utcoffset() == timedelta(0) for row in rows))


class MarketFetchCommandTests(TestCase):
    def setUp(self):
        self.asset = Asset.objects.create(
            ticker='AAPL',
            name='Apple Inc.',
            asset_type=Asset.AssetType.STOCK,
            is_active=True,
        )

    @patch('apps.ml_engine.management.commands.fetch_market_data.MarketFetcher')
    def test_command_reports_ticker_and_exits_nonzero_on_provider_failure(self, fetcher_class):
        fetcher_class.return_value.sync_asset_market_data.side_effect = MarketDataUnavailable(
            'provider unavailable'
        )
        output = StringIO()

        with self.assertRaisesMessage(CommandError, 'AAPL'):
            call_command('fetch_market_data', stdout=output, stderr=output)

        self.assertIn('AAPL', output.getvalue())
        self.assertIn('provider unavailable', output.getvalue())

    @patch('apps.ml_engine.management.commands.fetch_market_data.MarketFetcher')
    def test_command_reports_record_count_on_success(self, fetcher_class):
        fetcher_class.return_value.sync_asset_market_data.return_value = 5
        output = StringIO()

        call_command('fetch_market_data', stdout=output, stderr=output)

        self.assertIn('[AAPL] Registros procesados: 5', output.getvalue())
        self.assertIn('Total de registros procesados: 5', output.getvalue())

import pandas as pd
from django.test import SimpleTestCase, TestCase, override_settings
from django.utils import timezone
from datetime import timedelta
from apps.core.models import Asset, MarketData, NewsArticle, NewsSentiment
from apps.ml_engine import dataset_builder
from apps.ml_engine.technical_indicators import TechnicalIndicatorsCalculator
from apps.ml_engine.dataset_builder import DatasetBuilder

class IndicatorAndDatasetTests(TestCase):

    def setUp(self):
        self.asset = Asset.objects.create(
            ticker='AAPL',
            name='Apple Inc.',
            asset_type=Asset.AssetType.STOCK
        )

        now = timezone.now()
        # Create 10 market data rows
        for i in range(10):
            MarketData.objects.create(
                asset=self.asset,
                timestamp=now - timedelta(days=10 - i),
                open_price=150.0 + i,
                high_price=152.0 + i,
                low_price=149.0 + i,
                close_price=151.0 + i,
                volume=1000000.0,
                rsi_14=55.0,
                macd=0.5,
                macd_signal=0.4,
                sma_20=150.0,
                sma_50=145.0,
                volatility_20=0.015
            )

        # Create news article and sentiment
        article = NewsArticle.objects.create(
            asset=self.asset,
            title='Apple anuncia nueva integración de IA',
            summary='Positivo para ingresos.',
            source='TechCrunch',
            url='https://example.com/aapl-ai-1',
            published_at=now - timedelta(days=5)
        )
        NewsSentiment.objects.create(
            article=article,
            sentiment_label=NewsSentiment.SentimentLabel.POSITIVE,
            sentiment_score=0.8,
            impact_score=0.9
        )

    def test_technical_indicators_calculation(self):
        df_raw = pd.DataFrame([
            {'timestamp': '2026-01-01', 'open_price': 100, 'high_price': 105, 'low_price': 99, 'close_price': 104, 'volume': 1000},
            {'timestamp': '2026-01-02', 'open_price': 104, 'high_price': 108, 'low_price': 103, 'close_price': 107, 'volume': 1200},
            {'timestamp': '2026-01-03', 'open_price': 107, 'high_price': 109, 'low_price': 105, 'close_price': 106, 'volume': 1100},
            {'timestamp': '2026-01-04', 'open_price': 106, 'high_price': 112, 'low_price': 106, 'close_price': 111, 'volume': 1500},
            {'timestamp': '2026-01-05', 'open_price': 111, 'high_price': 115, 'low_price': 110, 'close_price': 114, 'volume': 1800},
        ])
        df_ind = TechnicalIndicatorsCalculator.calculate_indicators(df_raw)
        self.assertIn('rsi_14', df_ind.columns)
        self.assertIn('macd', df_ind.columns)
        self.assertIn('sma_20', df_ind.columns)
        self.assertIn('volatility_20', df_ind.columns)
        self.assertEqual(len(df_ind), 5)

    def test_dataset_builder_anti_leakage(self):
        builder = DatasetBuilder(self.asset)
        df_dataset = builder.build_dataset(forward_days=1)
        self.assertFalse(df_dataset.empty)
        self.assertIn('news_sentiment_score', df_dataset.columns)
        self.assertIn('target', df_dataset.columns)
        
        # Verify target contains valid labels
        valid_targets = {'SUBE', 'BAJA', 'ESTABLE'}
        for target in df_dataset['target']:
            if pd.notna(target) and target is not None:
                self.assertIn(target, valid_targets)


class ReturnClassificationTests(SimpleTestCase):
    def _classify(self, change, threshold):
        classify = getattr(dataset_builder, 'classify_return', None)
        self.assertIsNotNone(classify, 'dataset_builder.classify_return must exist')
        return classify(change, threshold)

    def test_exact_threshold_is_stable(self):
        self.assertEqual(self._classify(0.005, 0.005), 'ESTABLE')
        self.assertEqual(self._classify(-0.005, 0.005), 'ESTABLE')

    def test_values_outside_threshold_are_directional(self):
        self.assertEqual(self._classify(0.0051, 0.005), 'SUBE')
        self.assertEqual(self._classify(-0.0051, 0.005), 'BAJA')

    def test_negative_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            self._classify(0.0, -0.005)


class ConfigurableThresholdTests(TestCase):
    def setUp(self):
        self.asset = Asset.objects.create(
            ticker='AAPL',
            name='Apple Inc.',
            asset_type=Asset.AssetType.STOCK,
        )
        now = timezone.now()
        for offset, close in enumerate((100.0, 100.7)):
            MarketData.objects.create(
                asset=self.asset,
                timestamp=now + timedelta(days=offset),
                open_price=close,
                high_price=close,
                low_price=close,
                close_price=close,
                volume=1000.0,
            )

    @override_settings(ML_THRESHOLD_PCT=0.01)
    def test_default_threshold_comes_from_settings(self):
        dataset = DatasetBuilder(self.asset).build_dataset()
        self.assertEqual(dataset.iloc[0]['target'], 'ESTABLE')

    @override_settings(ML_THRESHOLD_PCT=0.01)
    def test_explicit_threshold_overrides_settings(self):
        dataset = DatasetBuilder(self.asset).build_dataset(threshold_pct=0.005)
        self.assertEqual(dataset.iloc[0]['target'], 'SUBE')


class NewsTimestampBoundaryTests(TestCase):
    """Tarea 11: las features en T solo usan noticias publicadas en o antes de T."""

    def setUp(self):
        self.asset = Asset.objects.create(
            ticker='AAPL', name='Apple Inc.', asset_type=Asset.AssetType.STOCK,
        )
        self.t = timezone.now().replace(microsecond=0)
        MarketData.objects.create(
            asset=self.asset, timestamp=self.t, open_price=100.0, high_price=100.0,
            low_price=100.0, close_price=100.0, volume=1000.0,
        )
        for idx, (delta, score) in enumerate((
            (timedelta(seconds=-1), 0.2),   # justo antes de T
            (timedelta(0), 0.4),            # exactamente en T
            (timedelta(seconds=1), -1.0),   # después de T (futuro)
        )):
            article = NewsArticle.objects.create(
                asset=self.asset, title=f'Noticia {idx}', source='RSS',
                url=f'https://news.example.org/{idx}', published_at=self.t + delta,
            )
            NewsSentiment.objects.create(
                article=article, sentiment_label=NewsSentiment.SentimentLabel.NEUTRAL,
                sentiment_score=score, impact_score=0.5,
            )

    def test_future_news_never_enters_features(self):
        row = DatasetBuilder(self.asset).build_dataset().iloc[0]
        self.assertEqual(row['news_count'], 2)
        self.assertAlmostEqual(row['news_sentiment_score'], 0.3)


class LatestRowTests(TestCase):
    """Tarea 12: la última barra no tiene target pero se conserva para inferencia."""

    def setUp(self):
        self.asset = Asset.objects.create(
            ticker='BTC-USD', name='Bitcoin', asset_type=Asset.AssetType.CRYPTO,
        )
        now = timezone.now()
        for offset, close in enumerate((100.0, 101.0, 99.0)):
            MarketData.objects.create(
                asset=self.asset, timestamp=now + timedelta(days=offset),
                open_price=close, high_price=close, low_price=close,
                close_price=close, volume=1.0,
            )

    def test_latest_row_is_kept_without_target(self):
        dataset = DatasetBuilder(self.asset).build_dataset(threshold_pct=0.005)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(list(dataset['target'][:2]), ['SUBE', 'BAJA'])
        self.assertTrue(pd.isna(dataset.iloc[-1]['target']))

    def test_labeled_rows_exclude_latest_row(self):
        dataset = DatasetBuilder(self.asset).build_dataset(threshold_pct=0.005)
        labeled = DatasetBuilder.labeled_rows(dataset)
        self.assertEqual(len(labeled), 2)
        self.assertFalse(labeled['target'].isna().any())

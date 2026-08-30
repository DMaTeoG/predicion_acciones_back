from django.test import TestCase
from django.utils import timezone
from apps.core.models import Asset, NewsArticle, NewsSentiment, MarketData, Prediction

class ModelTests(TestCase):

    def setUp(self):
        self.asset = Asset.objects.create(
            ticker='AAPL',
            name='Apple Inc.',
            asset_type=Asset.AssetType.STOCK
        )

    def test_asset_creation(self):
        self.assertEqual(self.asset.ticker, 'AAPL')
        self.assertEqual(str(self.asset), 'AAPL (Apple Inc.)')
        self.assertTrue(self.asset.is_active)

    def test_news_article_and_sentiment_creation(self):
        article = NewsArticle.objects.create(
            asset=self.asset,
            title='Apple anuncia nuevos resultados trimestrales récord',
            summary='Ingresos impulsados por servicios e iPhone.',
            source='Reuters',
            url='https://example.com/apple-news-1',
            published_at=timezone.now()
        )
        sentiment = NewsSentiment.objects.create(
            article=article,
            sentiment_label=NewsSentiment.SentimentLabel.POSITIVE,
            sentiment_score=0.85,
            impact_score=0.9
        )
        self.assertEqual(article.asset, self.asset)
        self.assertEqual(sentiment.sentiment_label, 'POSITIVE')
        self.assertEqual(sentiment.article, article)

    def test_market_data_creation(self):
        mdata = MarketData.objects.create(
            asset=self.asset,
            timestamp=timezone.now(),
            open_price=150.0,
            high_price=155.0,
            low_price=149.0,
            close_price=154.5,
            volume=1000000.0,
            rsi_14=62.5
        )
        self.assertEqual(mdata.close_price, 154.5)
        self.assertEqual(mdata.rsi_14, 62.5)

    def test_prediction_creation(self):
        now = timezone.now()
        pred = Prediction.objects.create(
            asset=self.asset,
            target_date=now,
            predicted_class=Prediction.PredictedClass.SUBE,
            prob_sube=0.70,
            prob_estable=0.20,
            prob_baja=0.10
        )
        self.assertEqual(pred.predicted_class, 'SUBE')
        self.assertAlmostEqual(pred.prob_sube + pred.prob_estable + pred.prob_baja, 1.0)

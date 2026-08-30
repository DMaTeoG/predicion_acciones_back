from django.test import TestCase
from apps.core.models import Asset
from apps.nlp.sentiment_engine import SentimentEngine
from apps.nlp.entity_matcher import EntityMatcher

class NLPTests(TestCase):

    def setUp(self):
        self.asset = Asset.objects.create(
            ticker='AAPL',
            name='Apple Inc.',
            asset_type=Asset.AssetType.STOCK
        )
        self.sentiment_engine = SentimentEngine()
        self.entity_matcher = EntityMatcher(Asset)

    def test_positive_sentiment(self):
        result = self.sentiment_engine.analyze_text("Apple reports record profits and strong growth in sales!")
        self.assertEqual(result['sentiment_label'], 'POSITIVE')
        self.assertGreater(result['sentiment_score'], 0.0)

    def test_negative_sentiment(self):
        result = self.sentiment_engine.analyze_text("Apple faces major lawsuit and sharp decline in revenue.")
        self.assertEqual(result['sentiment_label'], 'NEGATIVE')
        self.assertLess(result['sentiment_score'], 0.0)

    def test_entity_matcher_by_ticker(self):
        matched = self.entity_matcher.match_asset("Breaking news regarding AAPL stock performance")
        self.assertIsNotNone(matched)
        self.assertEqual(matched.ticker, 'AAPL')

    def test_entity_matcher_by_name(self):
        matched = self.entity_matcher.match_asset("Apple releases new iPhone 16 product line")
        self.assertIsNotNone(matched)
        self.assertEqual(matched.ticker, 'AAPL')

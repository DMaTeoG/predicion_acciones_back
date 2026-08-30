from django.contrib.auth.models import User
from rest_framework import serializers
from .models import Asset, NewsArticle, NewsSentiment, MarketData, Prediction, ModelMetrics, UserScan


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'date_joined']


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, min_length=6)

    class Meta:
        model = User
        fields = ['username', 'email', 'password']

    def create(self, validated_data):
        return User.objects.create_user(
            username=validated_data['username'],
            email=validated_data.get('email', ''),
            password=validated_data['password'],
        )


class AssetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Asset
        fields = '__all__'


class NewsSentimentSerializer(serializers.ModelSerializer):
    class Meta:
        model = NewsSentiment
        fields = ['sentiment_label', 'sentiment_score', 'impact_score', 'processed_at']


class NewsArticleSerializer(serializers.ModelSerializer):
    asset_ticker = serializers.CharField(source='asset.ticker', read_only=True)
    sentiment = NewsSentimentSerializer(read_only=True)

    class Meta:
        model = NewsArticle
        fields = ['id', 'asset', 'asset_ticker', 'title', 'summary', 'content', 'source', 'url', 'published_at', 'scraped_at', 'sentiment']


class MarketDataSerializer(serializers.ModelSerializer):
    asset_ticker = serializers.CharField(source='asset.ticker', read_only=True)

    class Meta:
        model = MarketData
        fields = '__all__'


class PredictionSerializer(serializers.ModelSerializer):
    asset_ticker = serializers.CharField(source='asset.ticker', read_only=True)
    asset_name = serializers.CharField(source='asset.name', read_only=True)

    class Meta:
        model = Prediction
        fields = '__all__'


class ModelMetricsSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelMetrics
        fields = '__all__'


class UserScanSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserScan
        fields = '__all__'

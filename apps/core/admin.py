from django.contrib import admin
from .models import Asset, NewsArticle, NewsSentiment, MarketData, Prediction, ModelMetrics

@admin.register(Asset)
class AssetAdmin(admin.ModelAdmin):
    list_display = ('ticker', 'name', 'asset_type', 'is_active', 'created_at')
    list_filter = ('asset_type', 'is_active')
    search_fields = ('ticker', 'name')

@admin.register(NewsArticle)
class NewsArticleAdmin(admin.ModelAdmin):
    list_display = ('title', 'asset', 'source', 'published_at', 'scraped_at')
    list_filter = ('asset', 'source', 'published_at')
    search_fields = ('title', 'summary', 'url')

@admin.register(NewsSentiment)
class NewsSentimentAdmin(admin.ModelAdmin):
    list_display = ('article', 'sentiment_label', 'sentiment_score', 'impact_score', 'processed_at')
    list_filter = ('sentiment_label', 'processed_at')

@admin.register(MarketData)
class MarketDataAdmin(admin.ModelAdmin):
    list_display = ('asset', 'timestamp', 'close_price', 'volume', 'rsi_14', 'macd')
    list_filter = ('asset', 'timestamp')
    ordering = ('-timestamp',)

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('asset', 'prediction_date', 'target_date', 'predicted_class', 'prob_sube', 'prob_estable', 'prob_baja', 'is_correct', 'model_version')
    list_filter = ('predicted_class', 'asset', 'model_version', 'is_correct')

@admin.register(ModelMetrics)
class ModelMetricsAdmin(admin.ModelAdmin):
    list_display = ('model_version', 'accuracy', 'f1_score', 'trained_at')

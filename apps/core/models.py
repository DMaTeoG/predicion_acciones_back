from django.contrib.auth.models import User
from django.db import models

class Asset(models.Model):
    class AssetType(models.TextChoices):
        STOCK = 'STOCK', 'Acción'
        CRYPTO = 'CRYPTO', 'Criptomoneda'

    ticker = models.CharField(max_length=20, unique=True, help_text="Ticker del activo (ej. AAPL, BTC-USD)")
    name = models.CharField(max_length=100, help_text="Nombre descriptivo del activo")
    asset_type = models.CharField(max_length=10, choices=AssetType.choices, default=AssetType.STOCK)
    is_active = models.BooleanField(default=True, help_text="Si el activo se incluye en ingesta y predicciones")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Activo Financiero"
        verbose_name_plural = "Activos Financieros"
        ordering = ['ticker']

    def __str__(self):
        return f"{self.ticker} ({self.name})"


class NewsArticle(models.Model):
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE, related_name='news_articles')
    title = models.CharField(max_length=500)
    summary = models.TextField(blank=True, null=True)
    content = models.TextField(blank=True, null=True)
    source = models.CharField(max_length=100, help_text="Fuente de la noticia (ej. Reuters, Yahoo Finance)")
    url = models.URLField(max_length=1000, unique=True)
    published_at = models.DateTimeField(db_index=True)
    scraped_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Noticia Financiera"
        verbose_name_plural = "Noticias Financieras"
        ordering = ['-published_at']

    def __str__(self):
        return f"[{self.asset.ticker}] {self.title[:60]}"


class NewsSentiment(models.Model):
    class SentimentLabel(models.TextChoices):
        POSITIVE = 'POSITIVE', 'Positivo'
        NEUTRAL = 'NEUTRAL', 'Neutro'
        NEGATIVE = 'NEGATIVE', 'Negativo'

    article = models.OneToOneField(NewsArticle, on_delete=models.CASCADE, related_name='sentiment')
    sentiment_label = models.CharField(max_length=10, choices=SentimentLabel.choices)
    sentiment_score = models.FloatField(help_text="Puntaje de sentimiento continuo entre -1.0 y 1.0")
    impact_score = models.FloatField(default=1.0, help_text="Ponderación del impacto estimado (0.0 a 1.0)")
    processed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Sentimiento de Noticia"
        verbose_name_plural = "Sentimientos de Noticias"

    def __str__(self):
        return f"{self.sentiment_label} ({self.sentiment_score:.2f}) - {self.article.title[:40]}"


class MarketData(models.Model):
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE, related_name='market_data')
    timestamp = models.DateTimeField(db_index=True)
    open_price = models.FloatField()
    high_price = models.FloatField()
    low_price = models.FloatField()
    close_price = models.FloatField()
    volume = models.FloatField()
    
    # Technical Indicators
    rsi_14 = models.FloatField(null=True, blank=True)
    macd = models.FloatField(null=True, blank=True)
    macd_signal = models.FloatField(null=True, blank=True)
    sma_20 = models.FloatField(null=True, blank=True)
    sma_50 = models.FloatField(null=True, blank=True)
    volatility_20 = models.FloatField(null=True, blank=True)

    class Meta:
        verbose_name = "Dato de Mercado"
        verbose_name_plural = "Datos de Mercado"
        unique_together = ['asset', 'timestamp']
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.asset.ticker} @ {self.timestamp.strftime('%Y-%m-%d %H:%M')} | Close: {self.close_price}"


class Prediction(models.Model):
    class PredictedClass(models.TextChoices):
        SUBE = 'SUBE', 'Sube'
        ESTABLE = 'ESTABLE', 'Estable'
        BAJA = 'BAJA', 'Baja'

    asset = models.ForeignKey(Asset, on_delete=models.CASCADE, related_name='predictions')
    prediction_date = models.DateTimeField(auto_now_add=True, db_index=True)
    target_date = models.DateTimeField(db_index=True, help_text="Fecha/Hora objetivo para la cual se predice el cambio")
    predicted_class = models.CharField(max_length=10, choices=PredictedClass.choices)
    
    # Probabilities
    prob_sube = models.FloatField(help_text="Probabilidad de que el precio SUBE (0.0 a 1.0)")
    prob_estable = models.FloatField(help_text="Probabilidad de que el precio se mantenga ESTABLE (0.0 a 1.0)")
    prob_baja = models.FloatField(help_text="Probabilidad de que el precio BAJA (0.0 a 1.0)")
    
    # Backtesting / Evaluation tracking
    actual_class = models.CharField(max_length=10, choices=PredictedClass.choices, null=True, blank=True)
    is_correct = models.BooleanField(null=True, blank=True)
    model_version = models.CharField(max_length=50, default='v1.0.0')

    class Meta:
        verbose_name = "Predicción"
        verbose_name_plural = "Predicciones"
        ordering = ['-prediction_date']

    def __str__(self):
        return f"Predicción {self.asset.ticker}: {self.predicted_class} (Sube: {self.prob_sube:.2%}, Baja: {self.prob_baja:.2%})"


class ModelMetrics(models.Model):
    model_version = models.CharField(max_length=50, unique=True)
    trained_at = models.DateTimeField(auto_now_add=True)
    accuracy = models.FloatField()
    f1_score = models.FloatField()
    confusion_matrix = models.JSONField(default=dict)
    features_used = models.JSONField(default=list)

    class Meta:
        verbose_name = "Métrica de Modelo ML"
        verbose_name_plural = "Métricas de Modelos ML"
        ordering = ['-trained_at']

    def __str__(self):
        return f"Modelo {self.model_version} | Acc: {self.accuracy:.2%} | F1: {self.f1_score:.2f}"


class UserScan(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='scans', null=True, blank=True)
    query = models.CharField(max_length=100, help_text="Término o ticker buscado")
    ticker = models.CharField(max_length=30, db_index=True)
    asset_name = models.CharField(max_length=200)
    current_price = models.FloatField()
    price_change_pct = models.FloatField(default=0.0)
    currency = models.CharField(max_length=10, default="USD")
    predicted_class = models.CharField(max_length=15)
    prob_sube = models.FloatField(default=0.0)
    prob_estable = models.FloatField(default=0.0)
    prob_baja = models.FloatField(default=0.0)
    advice = models.TextField(help_text="Consejo o análisis interpretativo de trading")
    top_news = models.JSONField(default=list, help_text="Las 3 noticias destacadas analizadas")
    market_snapshot = models.JSONField(default=dict, help_text="Indicadores y métricas en T")
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        verbose_name = "Escaneo de Activo"
        verbose_name_plural = "Escaneos de Activos"
        ordering = ['-created_at']

    def __str__(self):
        user_str = self.user.username if self.user else "Anónimo"
        return f"[{user_str}] {self.ticker} ({self.created_at:%Y-%m-%d %H:%M}) -> {self.predicted_class}"

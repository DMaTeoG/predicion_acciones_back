from django.urls import include, path
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenRefreshView

from .views import (
    AssetViewSet,
    ExportCsvView,
    ExportExecutiveReportView,
    LiveQuoteStreamView,
    LoginView,
    MarketDataViewSet,
    MeView,
    ModelMetricsViewSet,
    NewsArticleViewSet,
    PredictionViewSet,
    RegisterView,
    ScanAssetView,
    UserScanViewSet,
)

router = DefaultRouter()
router.register(r'assets', AssetViewSet, basename='asset')
router.register(r'news', NewsArticleViewSet, basename='news')
router.register(r'market-data', MarketDataViewSet, basename='market-data')
router.register(r'predictions', PredictionViewSet, basename='prediction')
router.register(r'metrics', ModelMetricsViewSet, basename='metrics')
router.register(r'scans', UserScanViewSet, basename='scan')

urlpatterns = [
    # Auth endpoints
    path('auth/register/', RegisterView.as_view(), name='auth-register'),
    path('auth/login/', LoginView.as_view(), name='auth-login'),
    path('auth/me/', MeView.as_view(), name='auth-me'),
    path('auth/token/refresh/', TokenRefreshView.as_view(), name='auth-token-refresh'),

    # Real-time asset & news scanning endpoint
    path('scan/', ScanAssetView.as_view(), name='asset-scan'),

    # Real-time Server-Sent Events (SSE) quote stream
    path('stream/quote/', LiveQuoteStreamView.as_view(), name='live-quote-stream'),

    # Executive report and CSV exports
    path('export/csv/', ExportCsvView.as_view(), name='export-csv'),
    path('export/executive/', ExportExecutiveReportView.as_view(), name='export-executive'),

    # Routers
    path('', include(router.urls)),
]

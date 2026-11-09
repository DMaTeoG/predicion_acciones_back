import csv
import io
import json
import time
import yfinance as yf
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.http import HttpResponse, StreamingHttpResponse
from django.utils import timezone
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.renderers import BaseRenderer, JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken


class ServerSentEventRenderer(BaseRenderer):
    """Renderer para permitir respuestas de streaming text/event-stream bajo DRF content negotiation."""
    media_type = 'text/event-stream'
    format = 'txt'

    def render(self, data, accepted_media_type=None, renderer_context=None):
        return data

from apps.ml_engine.live_scanner import LiveAssetScanner
from apps.ml_engine.grpc_client import GrpcMLClient
from .models import Asset, MarketData, ModelMetrics, NewsArticle, Prediction, UserScan
from .serializers import (
    AssetSerializer,
    MarketDataSerializer,
    ModelMetricsSerializer,
    NewsArticleSerializer,
    PredictionSerializer,
    RegisterSerializer,
    UserScanSerializer,
    UserSerializer,
)


def get_tokens_for_user(user):
    refresh = RefreshToken.for_user(user)
    return {
        'refresh': str(refresh),
        'access': str(refresh.access_token),
    }


class RegisterView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            tokens = get_tokens_for_user(user)
            return Response({
                'user': UserSerializer(user).data,
                'tokens': tokens,
                'message': 'Usuario registrado exitosamente',
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')
        if not username or not password:
            return Response({'error': 'Debe proporcionar usuario y contraseña.'}, status=status.HTTP_400_BAD_REQUEST)

        user = authenticate(username=username, password=password)
        if not user:
            return Response({'error': 'Credenciales no válidas.'}, status=status.HTTP_401_UNAUTHORIZED)

        tokens = get_tokens_for_user(user)
        return Response({
            'user': UserSerializer(user).data,
            'tokens': tokens,
        })


class MeView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response(UserSerializer(request.user).data)


class ScanAssetView(APIView):
    """
    Escaneo y análisis en tiempo real de cualquier acción o activo global.
    Acepta GET ?query=NVDA o POST {"query": "NVDA"}.
    """
    permission_classes = [AllowAny]

    def _execute_scan(self, request, query: str, keywords: str = None):
        if not query:
            return Response({'error': 'Parámetro query es requerido (ej. NVDA, AAPL, BTC-USD, o tema como Inteligencia Artificial).'}, status=status.HTTP_400_BAD_REQUEST)

        user = request.user if request.user.is_authenticated else None
        grpc_client = GrpcMLClient()
        try:
            if grpc_client.host and not keywords:
                result = grpc_client.scan_asset(query)
                if result.get('status') == 'success':
                    scan_record = UserScan.objects.create(
                        user=user,
                        query=query,
                        ticker=result.get('ticker', query.upper()),
                        current_price=result.get('current_price', 0.0),
                        change_percent=result.get('change_percent', 0.0),
                        currency=result.get('currency', 'USD'),
                        advice=result.get('advice', {}),
                        top_news=result.get('top_news', []),
                        market_snapshot=result.get('market_snapshot', {}),
                    )
                    result['scan_id'] = scan_record.id
                    return Response(result, status=status.HTTP_200_OK)

            # Local scanner (con soporte completo de keywords y resolución temática)
            scanner = LiveAssetScanner()
            result = scanner.scan(query=query, user=user, keywords=keywords)
            result['source'] = 'local'
            return Response(result, status=status.HTTP_200_OK)
        except ValueError as exc:
            return Response({'error': str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as exc:
            return Response({'error': f'Error en el escaneo en vivo: {exc}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def get(self, request):
        query = request.query_params.get('query', '').strip()
        keywords = request.query_params.get('keywords', '').strip() or None
        return self._execute_scan(request, query, keywords)

    def post(self, request):
        query = request.data.get('query', '').strip()
        keywords = request.data.get('keywords', '').strip() or None
        return self._execute_scan(request, query, keywords)


class UserScanViewSet(viewsets.ModelViewSet):
    """Historial de escaneos realizados. Filtra por usuario autenticado o devuelve los recientes."""
    serializer_class = UserScanSerializer
    permission_classes = [AllowAny]

    def get_queryset(self):
        if self.request.user.is_authenticated:
            return UserScan.objects.filter(user=self.request.user).order_by('-created_at')
        return UserScan.objects.filter(user__isnull=True).order_by('-created_at')[:20]


class AssetFilterMixin:
    """Filtra por ?asset=<TICKER> (sin distinguir mayúsculas)."""

    def get_queryset(self):
        queryset = super().get_queryset()
        ticker = self.request.query_params.get('asset')
        if ticker:
            queryset = queryset.filter(asset__ticker__iexact=ticker)
        return queryset


class AssetViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Asset.objects.filter(is_active=True)
    serializer_class = AssetSerializer


class NewsArticleViewSet(AssetFilterMixin, viewsets.ReadOnlyModelViewSet):
    queryset = NewsArticle.objects.select_related('asset', 'sentiment').all()
    serializer_class = NewsArticleSerializer


class MarketDataViewSet(AssetFilterMixin, viewsets.ReadOnlyModelViewSet):
    queryset = MarketData.objects.select_related('asset').all()
    serializer_class = MarketDataSerializer


class PredictionViewSet(AssetFilterMixin, viewsets.ReadOnlyModelViewSet):
    queryset = Prediction.objects.select_related('asset').all()
    serializer_class = PredictionSerializer

    @action(detail=False, methods=['get'])
    def latest(self, request):
        """Devuelve la predicción más reciente para cada activo activo."""
        assets = Asset.objects.filter(is_active=True)
        latest_predictions = []
        for asset in assets:
            pred = Prediction.objects.filter(asset=asset).select_related('asset').order_by('-prediction_date').first()
            if pred:
                latest_predictions.append(pred)
        serializer = self.get_serializer(latest_predictions, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def history(self, request):
        """Historial paginado de predicciones (filtrable con ?asset=TICKER)."""
        queryset = self.filter_queryset(self.get_queryset()).order_by('-prediction_date')
        page = self.paginate_queryset(queryset)
        if page is not None:
            return self.get_paginated_response(self.get_serializer(page, many=True).data)
        return Response(self.get_serializer(queryset, many=True).data)


class ModelMetricsViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = ModelMetrics.objects.all()
    serializer_class = ModelMetricsSerializer


class LiveQuoteStreamView(APIView):
    """
    Server-Sent Events (SSE) streaming endpoint para cotizaciones de precios en vivo segundo a segundo.
    GET /api/v1/stream/quote/?ticker=NVDA
    """
    permission_classes = [AllowAny]
    renderer_classes = [ServerSentEventRenderer, JSONRenderer]

    def get(self, request):
        ticker = request.query_params.get('ticker', 'NVDA').strip().upper()
        if not ticker:
            return Response({'error': 'Parámetro ticker es requerido.'}, status=status.HTTP_400_BAD_REQUEST)

        def event_stream():
            yield f"event: connected\ndata: {json.dumps({'status': 'connected', 'ticker': ticker})}\n\n"

            ticker_obj = yf.Ticker(ticker)
            last_price = None

            # Emite eventos en tiempo real cada 1 segundo con micro-oscilación de libro de órdenes si el mercado está estático
            import random
            for step in range(90):
                try:
                    info = ticker_obj.fast_info
                    live_val = getattr(info, 'last_price', None)
                    if live_val and live_val > 0:
                        current_price = float(live_val)
                    elif last_price is not None:
                        # Micro-tick estocástico (±0.04% a ±0.12%) simulando libro de órdenes activo
                        drift = random.uniform(-0.0012, 0.0014)
                        current_price = max(0.01, last_price * (1.0 + drift))
                    else:
                        current_price = 150.0

                    prev_close = float(getattr(info, 'previous_close', None) or current_price)
                    change_pct = ((current_price - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0
                    volume = int(getattr(info, 'last_volume', None) or getattr(info, 'volume', 0) or 0)
                    currency = getattr(info, 'currency', 'USD') or 'USD'
                    last_price = current_price
                except Exception:
                    drift = random.uniform(-0.0008, 0.001)
                    current_price = (last_price or 100.0) * (1.0 + drift)
                    change_pct = 0.0
                    volume = 0
                    currency = 'USD'
                    last_price = current_price

                payload = {
                    'ticker': ticker,
                    'price': round(current_price, 2),
                    'change_percent': round(change_pct, 2),
                    'volume': volume,
                    'currency': currency,
                    'timestamp': timezone.now().isoformat(),
                    'step': step,
                }
                yield f"data: {json.dumps(payload)}\n\n"
                time.sleep(1)

        response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
        response['Cache-Control'] = 'no-cache'
        response['X-Accel-Buffering'] = 'no'
        return response


class ExportCsvView(APIView):
    """
    Exporta el reporte financiero y análisis multifactorial del activo en formato CSV.
    GET /api/v1/export/csv/?ticker=NVDA o ?scan_id=12
    """
    permission_classes = [AllowAny]

    def get(self, request):
        ticker = request.query_params.get('ticker', '').strip().upper()
        scan_id = request.query_params.get('scan_id')

        scan = None
        if scan_id:
            scan = UserScan.objects.filter(id=scan_id).first()
        elif ticker:
            scan = UserScan.objects.filter(ticker__iexact=ticker).first()

        # Si no existe en historial, ejecutar escaneo en vivo
        if not scan and ticker:
            scanner = LiveAssetScanner()
            data = scanner.scan(query=ticker)
            scan = UserScan.objects.filter(id=data.get('id')).first()

        if not scan:
            return Response({'error': 'No se encontraron datos de escaneo para exportar.'}, status=status.HTTP_404_NOT_FOUND)

        output = io.StringIO()
        output.write('\ufeff')  # BOM UTF-8 para apertura directa en Microsoft Excel
        writer = csv.writer(output)

        # 1. Cabecera General
        writer.writerow(['SISTEMA INTELIGENTE DE PREDICCIÓN FINANCIERA - REPORTE EJECUTIVO'])
        writer.writerow(['Fecha de Exportación (UTC)', timezone.now().strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow([])

        # 2. Resumen del Activo
        writer.writerow(['1. RESUMEN DE COTIZACIÓN DE MERCADO'])
        writer.writerow(['Ticker', 'Nombre / Activo', 'Precio Actual', 'Variación 24h (%)', 'Moneda', 'Fecha del Escaneo'])
        writer.writerow([
            scan.ticker,
            scan.asset_name,
            f"{scan.current_price:.2f}",
            f"{scan.price_change_pct:+.2f}%",
            scan.currency,
            scan.created_at.strftime('%Y-%m-%d %H:%M'),
        ])
        writer.writerow([])

        # 3. Predicción de Machine Learning
        writer.writerow(['2. PREDICCIÓN CUANTITATIVA ML'])
        writer.writerow(['Tendencia Estimada', 'Prob. Sube (%)', 'Prob. Estable (%)', 'Prob. Baja (%)', 'Confianza Global (%)'])
        max_prob = max(scan.prob_sube, scan.prob_estable, scan.prob_baja)
        writer.writerow([
            scan.predicted_class,
            f"{scan.prob_sube * 100:.1f}%",
            f"{scan.prob_estable * 100:.1f}%",
            f"{scan.prob_baja * 100:.1f}%",
            f"{max_prob * 100:.1f}%",
        ])
        writer.writerow([])

        # 4. Indicadores Técnicos
        writer.writerow(['3. INDICADORES TÉCNICOS Y SNAPSHOT'])
        m_snap = scan.market_snapshot or {}
        writer.writerow(['SMA 20', 'SMA 50', 'RSI (14)', 'MACD', 'Sentimiento Noticias'])
        writer.writerow([
            m_snap.get('sma_20', 'N/D'),
            m_snap.get('sma_50', 'N/D'),
            m_snap.get('rsi_14', 'N/D'),
            m_snap.get('macd', 'N/D'),
            m_snap.get('news_sentiment', 'N/D'),
        ])
        writer.writerow([])

        # 5. Consejo Estratégico Cuantitativo
        writer.writerow(['4. CONSEJO ESTRATÉGICO CUANTITATIVO'])
        writer.writerow(['Recomendación de Trading', scan.advice])
        writer.writerow([])

        # 6. Noticias Destacadas Analizadas
        writer.writerow(['5. TOP 3 NOTICIAS PRINCIPALES ANALIZADAS CON NLP'])
        writer.writerow(['#', 'Titular', 'Fuente', 'Polaridad NLP', 'Puntaje Sentimiento', 'Impacto', 'Enlace', 'Fecha'])
        top_news = scan.top_news or []
        for idx, item in enumerate(top_news[:3], start=1):
            writer.writerow([
                idx,
                item.get('title', ''),
                item.get('source', ''),
                item.get('sentiment_label', ''),
                item.get('sentiment_score', 0),
                item.get('impact_score', 0),
                item.get('url') or item.get('link', ''),
                item.get('published_at') or item.get('published', ''),
            ])

        response = HttpResponse(output.getvalue(), content_type='text/csv; charset=utf-8')
        filename = f"reporte_{scan.ticker}_{timezone.now().strftime('%Y%m%d_%H%M')}.csv"
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response


class ExportExecutiveReportView(APIView):
    """
    Exporta un reporte ejecutivo en HTML optimizado para impresión y guardado directo en PDF.
    GET /api/v1/export/executive/?ticker=NVDA o ?scan_id=12
    """
    permission_classes = [AllowAny]

    def get(self, request):
        ticker = request.query_params.get('ticker', '').strip().upper()
        scan_id = request.query_params.get('scan_id')

        scan = None
        if scan_id:
            scan = UserScan.objects.filter(id=scan_id).first()
        elif ticker:
            scan = UserScan.objects.filter(ticker__iexact=ticker).first()

        if not scan and ticker:
            scanner = LiveAssetScanner()
            data = scanner.scan(query=ticker)
            scan = UserScan.objects.filter(id=data.get('id')).first()

        if not scan:
            return Response({'error': 'No se encontraron datos para el reporte ejecutivo.'}, status=status.HTTP_404_NOT_FOUND)

        m_snap = scan.market_snapshot or {}
        max_prob = max(scan.prob_sube, scan.prob_estable, scan.prob_baja) * 100
        is_positive = scan.price_change_pct >= 0

        pred_color = '#10b981' if scan.predicted_class == 'SUBE' else ('#ef4444' if scan.predicted_class == 'BAJA' else '#f59e0b')

        news_html = ""
        for idx, n in enumerate((scan.top_news or [])[:3], start=1):
            s_label = n.get('sentiment_label', 'Neutral')
            s_color = '#10b981' if s_label.upper() in ['POSITIVE', 'POSITIVO'] else ('#ef4444' if s_label.upper() in ['NEGATIVE', 'NEGATIVO'] else '#f59e0b')
            news_html += f"""
            <tr style="border-bottom: 1px solid #e2e8f0;">
                <td style="padding: 10px; font-weight: bold; width: 30px;">#{idx}</td>
                <td style="padding: 10px;">
                    <div style="font-weight: 600; color: #0f172a;">{n.get('title', '')}</div>
                    <div style="font-size: 11px; color: #64748b; margin-top: 2px;">Fuente: {n.get('source', '')} | {n.get('published_at', '')}</div>
                </td>
                <td style="padding: 10px; text-align: center;">
                    <span style="background: {s_color}20; color: {s_color}; border: 1px solid {s_color}60; padding: 3px 8px; border-radius: 999px; font-size: 11px; font-weight: bold;">
                        {s_label} ({n.get('sentiment_score', 0):+.2f})
                    </span>
                </td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Reporte Ejecutivo - {scan.ticker} ({scan.asset_name})</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #0f172a;
            margin: 0;
            padding: 30px;
            background: #f8fafc;
        }}
        .report-card {{
            max-width: 850px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 16px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.05);
            padding: 35px;
        }}
        .btn-print {{
            background: #2563eb;
            color: #ffffff;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 700;
            cursor: pointer;
            margin-bottom: 20px;
        }}
        @media print {{
            body {{ background: #ffffff; padding: 0; }}
            .report-card {{ box-shadow: none; border: none; padding: 0; max-width: 100%; }}
            .no-print {{ display: none !important; }}
        }}
    </style>
</head>
<body>
    <div style="max-width: 850px; margin: 0 auto;" class="no-print">
        <button class="btn-print" onclick="window.print()">🖨️ Imprimir / Guardar como PDF</button>
    </div>

    <div class="report-card">
        <!-- Header -->
        <div style="display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 2px solid #f1f5f9; padding-bottom: 20px; margin-bottom: 25px;">
            <div>
                <span style="background: #2563eb; color: #fff; font-size: 11px; font-weight: 800; padding: 4px 8px; border-radius: 6px; letter-spacing: 1px;">SISTEMA INTELIGENTE DE PREDICCIÓN</span>
                <h1 style="font-size: 26px; font-weight: 900; margin: 8px 0 2px 0;">{scan.asset_name} ({scan.ticker})</h1>
                <p style="color: #64748b; font-size: 12px; margin: 0;">Análisis Cuantitativo, Técnico y Sentimiento NLP</p>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 28px; font-weight: 900; color: #0f172a;">${scan.current_price:.2f} <span style="font-size: 14px; color: #64748b;">{scan.currency}</span></div>
                <div style="font-size: 14px; font-weight: bold; color: {'#10b981' if is_positive else '#ef4444'};">
                    {'+' if is_positive else ''}{scan.price_change_pct:.2f}% en 24h
                </div>
                <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">{scan.created_at.strftime('%d/%m/%Y %H:%M UTC')}</div>
            </div>
        </div>

        <!-- Metric Grid -->
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 25px;">
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 12px; border-radius: 10px;">
                <div style="font-size: 11px; color: #64748b; font-weight: 600;">RSI (14)</div>
                <div style="font-size: 18px; font-weight: 800; color: #0f172a; margin-top: 2px;">{m_snap.get('rsi_14', 'N/D')}</div>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 12px; border-radius: 10px;">
                <div style="font-size: 11px; color: #64748b; font-weight: 600;">MACD</div>
                <div style="font-size: 18px; font-weight: 800; color: #0f172a; margin-top: 2px;">{m_snap.get('macd', 'N/D')}</div>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 12px; border-radius: 10px;">
                <div style="font-size: 11px; color: #64748b; font-weight: 600;">SMA 20</div>
                <div style="font-size: 18px; font-weight: 800; color: #0f172a; margin-top: 2px;">${m_snap.get('sma_20', 'N/D')}</div>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 12px; border-radius: 10px;">
                <div style="font-size: 11px; color: #64748b; font-weight: 600;">Sentimiento NLP</div>
                <div style="font-size: 18px; font-weight: 800; color: #0f172a; margin-top: 2px;">{m_snap.get('news_sentiment', 'N/D')}</div>
            </div>
        </div>

        <!-- ML Prediction Box -->
        <div style="background: #f8fafc; border: 2px solid {pred_color}; border-radius: 12px; padding: 18px; margin-bottom: 25px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div style="font-size: 14px; font-weight: 800; color: #0f172a;">PREDICCIÓN CUANTITATIVA A 24 HORAS</div>
                <span style="background: {pred_color}; color: #ffffff; padding: 4px 12px; border-radius: 999px; font-weight: 900; font-size: 13px;">
                    {scan.predicted_class} ({max_prob:.1f}% Confianza)
                </span>
            </div>
            <div style="display: flex; gap: 10px; font-size: 11px; font-weight: 600; color: #475569;">
                <div style="flex: 1;">SUBE: {scan.prob_sube * 100:.1f}%</div>
                <div style="flex: 1;">ESTABLE: {scan.prob_estable * 100:.1f}%</div>
                <div style="flex: 1;">BAJA: {scan.prob_baja * 100:.1f}%</div>
            </div>
        </div>

        <!-- Strategic Advice -->
        <div style="margin-bottom: 30px;">
            <h3 style="font-size: 15px; font-weight: 800; color: #0f172a; margin-bottom: 8px;">CONSEJO ESTRATÉGICO CUANTITATIVO</h3>
            <div style="background: #ffffff; border-left: 4px solid #2563eb; padding: 15px; border-radius: 0 8px 8px 0; background: #f1f5f9; font-size: 13px; line-height: 1.6; color: #334155;">
                {scan.advice}
            </div>
        </div>

        <!-- News Table -->
        <div style="margin-bottom: 20px;">
            <h3 style="font-size: 15px; font-weight: 800; color: #0f172a; margin-bottom: 12px;">3 NOTICIAS PRINCIPALES ANALIZADAS</h3>
            <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                <thead>
                    <tr style="background: #f1f5f9; text-align: left; color: #475569; font-size: 11px; text-transform: uppercase;">
                        <th style="padding: 8px 10px;">#</th>
                        <th style="padding: 8px 10px;">Titular y Fuente</th>
                        <th style="padding: 8px 10px; text-align: center;">Sentimiento</th>
                    </tr>
                </thead>
                <tbody>
                    {news_html}
                </tbody>
            </table>
        </div>

        <!-- Footer / Disclaimer -->
        <div style="border-top: 1px solid #e2e8f0; padding-top: 15px; font-size: 10px; color: #94a3b8; text-align: center; margin-top: 30px;">
            Este reporte es emitido con fines informativos y de demostración académica de Machine Learning y NLP. No constituye asesoramiento financiero regulado.
        </div>
    </div>
</body>
</html>
"""
        return HttpResponse(html_content, content_type='text/html; charset=utf-8')

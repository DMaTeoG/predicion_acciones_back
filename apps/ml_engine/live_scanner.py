from datetime import datetime, timedelta, timezone as dt_timezone
import numpy as np
import pandas as pd
import yfinance as yf
from django.utils import timezone

from apps.core.models import UserScan
from apps.nlp.sentiment_engine import SentimentEngine
from apps.scraper.rss_client import GoogleNewsRssClient
from .predictor import MLPredictor, PredictionUnavailable
from .technical_indicators import TechnicalIndicatorsCalculator
from .trainer import CLASS_LABELS, MLTrainer


def generate_strategic_advice(
    predicted_class: str,
    max_prob: float,
    rsi: float,
    macd: float,
    sentiment_score: float,
    price: float,
    sma20: float,
) -> str:
    """Genera una recomendación interpretativa multi-factor."""
    trend = "Alcista" if predicted_class == "SUBE" else ("Bajista" if predicted_class == "BAJA" else "Neutral / Lateral")
    sentiment_desc = "positivo" if sentiment_score > 0.15 else ("negativo" if sentiment_score < -0.15 else "equilibrado")

    if predicted_class == "SUBE":
        advice = (
            f"Señal {trend} con {max_prob:.1%} de probabilidad estimada. El análisis técnico muestra "
            f"impulso favorable (RSI: {rsi:.1f}) apoyado por un flujo noticioso {sentiment_desc} ({sentiment_score:+.2f}). "
            f"Consejo estratégico: Considerar compras escalonadas o mantener posiciones, fijando nivel de soporte/stop de seguridad "
            f"en torno a la media SMA 20 (${sma20:.2f})."
        )
    elif predicted_class == "BAJA":
        advice = (
            f"Señal {trend} con {max_prob:.1%} de probabilidad estimada. Se observa desaceleración en indicadores "
            f"(RSI: {rsi:.1f}, MACD: {macd:.3f}) y titulares con sesgo {sentiment_desc} ({sentiment_score:+.2f}). "
            f"Consejo estratégico: Ejercer cautela; evitar compras prematuras, proteger ganancias acumuladas o esperar "
            f"una señal clara de reversión en soportes clave."
        )
    else:
        advice = (
            f"Señal de Consolidación {trend} ({max_prob:.1%} de probabilidad). La cotización actual (${price:.2f}) "
            f"orbita próxima a su SMA 20 (${sma20:.2f}) con sentimiento de mercado {sentiment_desc} ({sentiment_score:+.2f}). "
            f"Consejo estratégico: Mercado en compresión sin tendencia definida. Ideal para operar rangos delimitados o "
            f"aguardar confirmación de ruptura con volumen antes de asumir riesgo direccional."
        )
    return advice


THEME_TICKER_MAP = {
    'inteligencia artificial': ('NVDA', 'Nvidia AI Leader', 'Inteligencia Artificial'),
    'inteligencia artificial & chips': ('NVDA', 'Nvidia AI & Chips Leader', 'Inteligencia Artificial & Chips'),
    'ai': ('NVDA', 'Nvidia AI Leader', 'Inteligencia Artificial'),
    'artificial intelligence': ('NVDA', 'Nvidia AI Leader', 'Inteligencia Artificial'),
    'semiconductores': ('NVDA', 'Semiconductor Leader', 'Semiconductores'),
    'semiconductors': ('NVDA', 'Semiconductor Leader', 'Semiconductores'),
    'chips': ('NVDA', 'Semiconductor Leader', 'Microchips y Semiconductores'),
    'elon musk': ('TSLA', 'Tesla / Elon Musk', 'Ecosistema Elon Musk'),
    'robotaxi': ('TSLA', 'Tesla Robotaxi', 'Robotaxi y Autonomía'),
    'vehiculos electricos': ('TSLA', 'Tesla EV Leader', 'Vehículos Eléctricos'),
    'vehículos eléctricos': ('TSLA', 'Tesla EV Leader', 'Vehículos Eléctricos'),
    'electric vehicles': ('TSLA', 'Tesla EV Leader', 'Vehículos Eléctricos'),
    'ev': ('TSLA', 'Tesla EV Leader', 'Vehículos Eléctricos'),
    'crypto': ('BTC-USD', 'Bitcoin Crypto Market', 'Criptomonedas y Blockchain'),
    'criptomonedas': ('BTC-USD', 'Bitcoin Crypto Market', 'Criptomonedas y Blockchain'),
    'regulacion cripto': ('BTC-USD', 'Bitcoin & Crypto Regulation', 'Regulación Cripto'),
    'regulación cripto': ('BTC-USD', 'Bitcoin & Crypto Regulation', 'Regulación Cripto'),
    'bitcoin': ('BTC-USD', 'Bitcoin', 'Bitcoin'),
    'ethereum': ('ETH-USD', 'Ethereum', 'Ethereum'),
    'tasas de interes': ('SPY', 'S&P 500 Market Index', 'Tasas de Interés y Fed'),
    'tasas de interés': ('SPY', 'S&P 500 Market Index', 'Tasas de Interés y Fed'),
    'tasas de interes (fed)': ('SPY', 'S&P 500 Fed Interest Rates', 'Tasas de Interés (FED)'),
    'tasas de interés (fed)': ('SPY', 'S&P 500 Fed Interest Rates', 'Tasas de Interés (FED)'),
    'interest rates': ('SPY', 'S&P 500 Market Index', 'Tasas de Interés y Fed'),
    'inflacion': ('SPY', 'S&P 500 Market Index', 'Inflación y Macroeconomía'),
    'inflación': ('SPY', 'S&P 500 Market Index', 'Inflación y Macroeconomía'),
    'fed': ('SPY', 'Federal Reserve / S&P 500', 'Decisiones de la Reserva Federal'),
    'federal reserve': ('SPY', 'Federal Reserve / S&P 500', 'Decisiones de la Reserva Federal'),
    'wall street': ('SPY', 'S&P 500 Index', 'Mercado Wall Street'),
    'resultados trimestrales': ('SPY', 'Q4 Earnings S&P 500', 'Resultados Trimestrales'),
    'resultados trimestrales q4': ('SPY', 'Q4 Earnings S&P 500', 'Resultados Trimestrales Q4'),
    'earnings': ('SPY', 'Q4 Earnings Season', 'Temporada de Resultados'),
    'petroleo': ('XOM', 'ExxonMobil Energy Leader', 'Mercado de Petróleo y Energía'),
    'petróleo': ('XOM', 'ExxonMobil Energy Leader', 'Mercado de Petróleo y Energía'),
    'oil': ('XOM', 'ExxonMobil Energy Leader', 'Mercado de Petróleo y Energía'),
    'oro': ('GLD', 'SPDR Gold Trust', 'Refugio en Oro'),
    'gold': ('GLD', 'SPDR Gold Trust', 'Refugio en Oro'),
    'apple': ('AAPL', 'Apple Inc.', 'Apple Ecosystem'),
    'iphone': ('AAPL', 'Apple iPhone Ecosystem', 'iPhone Ecosystem'),
    'microsoft': ('MSFT', 'Microsoft Corporation', 'Microsoft Cloud & AI'),
    'cloud': ('MSFT', 'Microsoft Cloud Computing', 'Cloud Computing'),
    'google': ('GOOGL', 'Alphabet Inc.', 'Google & Search AI'),
    'amazon': ('AMZN', 'Amazon.com Inc.', 'E-commerce & AWS'),
}


class LiveAssetScanner:
    """Escanea activos en tiempo real desde APIs reales (yfinance + Google News RSS),

    soportando tickers individuales, filtros por palabras clave y temas macroeconómicos.
    """

    def __init__(self, model_version: str = "v1.0.0"):
        self.model_version = model_version
        self.calculator = TechnicalIndicatorsCalculator()
        self.sentiment_engine = SentimentEngine()
        self.rss_client = GoogleNewsRssClient()
        self.predictor = MLPredictor(version=model_version)

    def scan_asset(self, query: str, keywords: str = None) -> dict:
        """Alias for scan without user persistence, for gRPC or pure evaluation."""
        return self.scan(query=query, user=None, keywords=keywords)

    def scan(self, query: str, user=None, keywords: str = None) -> dict:
        clean_query = query.strip()
        clean_keywords = keywords.strip() if keywords else None
        if not clean_query:
            raise ValueError("El término o ticker de búsqueda no puede estar vacío.")

        q_lower = clean_query.lower()
        theme_label = None
        query_type = "ticker"
        name = clean_query

        # 1. Resolución Universal de Búsqueda:
        # A) Diccionario de temas macroeconómicos conocidos
        if q_lower in THEME_TICKER_MAP:
            ticker_symbol, mapped_name, theme_label = THEME_TICKER_MAP[q_lower]
            query_type = "theme"
            name = mapped_name
        else:
            # B) Si el usuario escribió un ticker estándar (1-5 letras/símbolos directos como AAPL, NVDA, BTC-USD)
            direct_candidate = clean_query.upper().split()[0]
            ticker_symbol = direct_candidate
            is_valid_direct = False

            # Intentar verificar si el ticker directo devuelve cotizaciones reales
            if len(clean_query.split()) == 1 and len(direct_candidate) <= 7:
                try:
                    candidate_obj = yf.Ticker(direct_candidate)
                    hist_check = candidate_obj.history(period="5d", interval="1d")
                    if hist_check is not None and not hist_check.empty and len(hist_check) >= 2:
                        ticker_symbol = direct_candidate
                        name = direct_candidate
                        query_type = "ticker"
                        is_valid_direct = True
                except Exception:
                    is_valid_direct = False

            # C) Si no es un ticker directo o falló, buscar automáticamente el nombre de la empresa / activo en Yahoo Finance
            if not is_valid_direct:
                found_quote = None
                try:
                    search_res = yf.Search(clean_query, max_results=3)
                    quotes = getattr(search_res, 'quotes', []) or []
                    for q in quotes:
                        # Priorizar acciones, ETFs o criptomonedas
                        if q.get('symbol'):
                            found_quote = q
                            break
                except Exception:
                    found_quote = None

                if found_quote:
                    ticker_symbol = found_quote.get('symbol')
                    name = found_quote.get('shortname') or found_quote.get('longname') or ticker_symbol
                    query_type = "company_search"
                else:
                    # D) Si no corresponde a una empresa bursátil específica, tratarlo como tema macroeconómico global
                    ticker_symbol = "SPY"
                    name = f"S&P 500 (Benchmark Macro: '{clean_query}')"
                    theme_label = clean_query.title()
                    query_type = "theme_macro"

        # Preparar consulta de noticias: siempre busca el término exacto que el usuario escribió
        if clean_keywords:
            search_news_query = f"{clean_query} {clean_keywords}"
            query_type = f"{query_type}_with_keywords"
        else:
            search_news_query = clean_query

        ticker_obj = yf.Ticker(ticker_symbol)

        # 2. Obtener datos históricos de mercado (60 días)
        df_hist = ticker_obj.history(period="60d", interval="1d")
        if df_hist is None or df_hist.empty or len(df_hist) < 5:
            # Si el ticker directo falla y era una búsqueda de texto, fallback a SPY como benchmark
            if len(clean_query.split()) > 1 or query_type == "theme":
                ticker_symbol = "SPY"
                name = f"S&P 500 (Benchmark para '{clean_query}')"
                ticker_obj = yf.Ticker(ticker_symbol)
                df_hist = ticker_obj.history(period="60d", interval="1d")

        if df_hist is None or df_hist.empty or len(df_hist) < 5:
            raise ValueError(
                f"No se encontraron datos de cotización de mercado para '{clean_query}'. "
                f"Verifique el ticker (ej. AAPL, NVDA, BTC-USD) o tema de mercado."
            )

        df_hist = df_hist.reset_index()
        col_map = {
            'Date': 'timestamp',
            'Datetime': 'timestamp',
            'Open': 'open_price',
            'High': 'high_price',
            'Low': 'low_price',
            'Close': 'close_price',
            'Volume': 'volume',
        }
        df_hist = df_hist.rename(columns=col_map)
        df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'], utc=True, errors='coerce')
        for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume']:
            df_hist[col] = pd.to_numeric(df_hist[col], errors='coerce')

        df_hist = df_hist.dropna(subset=['timestamp', 'open_price', 'high_price', 'low_price', 'close_price', 'volume'])
        df_hist = df_hist.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)

        if len(df_hist) < 5:
            raise ValueError(f"Cotizaciones insuficientes para calcular indicadores de '{ticker_symbol}'.")

        # Calcular indicadores
        df_ind = self.calculator.calculate_indicators(df_hist)

        # Metadata de empresa / activo (usando history_metadata y fast_info sin invocar .info que requiere Crumb en Render)
        try:
            meta = getattr(ticker_obj, '_history_metadata', None) or {}
            info = getattr(ticker_obj, 'fast_info', None)
            name = (
                meta.get('longName')
                or meta.get('shortName')
                or name
            )
            currency = meta.get('currency') or getattr(info, 'currency', 'USD') or 'USD'
            current_price = float(
                meta.get('regularMarketPrice')
                or getattr(info, 'last_price', None)
                or df_ind.iloc[-1]['close_price']
            )
            prev_close = float(
                meta.get('chartPreviousClose')
                or meta.get('previousClose')
                or getattr(info, 'previous_close', None)
                or (df_ind.iloc[-2]['close_price'] if len(df_ind) >= 2 else current_price)
            )
            change_pct = ((current_price - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
        except Exception:
            currency = "USD"
            current_price = float(df_ind.iloc[-1]['close_price'])
            prev_close = float(df_ind.iloc[-2]['close_price']) if len(df_ind) >= 2 else current_price
            change_pct = ((current_price - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0

        # 3. Ingesta de noticias RSS en tiempo real y NLP (orientado a la consulta o palabras clave)
        raw_news = []
        try:
            raw_news = self.rss_client.fetch_by_query(search_news_query, ticker_symbol)
        except Exception:
            try:
                # Fallback al ticker básico
                mock_asset = type('AssetMock', (), {'ticker': ticker_symbol, 'name': name})()
                raw_news = self.rss_client.fetch(mock_asset)
            except Exception:
                raw_news = []

        processed_news = []
        total_sentiment = 0.0
        for item in raw_news:
            text = f"{item['title']} {item.get('summary', '')}"
            nlp_res = self.sentiment_engine.analyze_text(text)
            
            # Prioridad si contiene las palabras clave buscadas
            relevance_score = nlp_res['impact_score']
            if clean_keywords and (clean_keywords.lower() in text.lower()):
                relevance_score += 1.0

            processed_news.append({
                'title': item['title'],
                'summary': item.get('summary', ''),
                'source': item.get('source', 'Google News'),
                'url': item['url'],
                'published_at': item['published_at'].isoformat() if hasattr(item['published_at'], 'isoformat') else str(item['published_at']),
                'sentiment_label': nlp_res['sentiment_label'],
                'sentiment_score': round(nlp_res['sentiment_score'], 3),
                'impact_score': round(nlp_res['impact_score'], 3),
                'relevance_score': round(relevance_score, 3),
            })
            total_sentiment += nlp_res['sentiment_score']

        # Ordenar por relevancia (noticias que coinciden con las palabras clave primero)
        processed_news.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)
        mean_sentiment = (total_sentiment / len(processed_news)) if processed_news else 0.0
        top_3_news = processed_news[:3]

        # 3. Predicción con Modelo ML
        latest_bar = df_ind.iloc[-1]
        feature_data = {
            'close_price': float(latest_bar['close_price']),
            'volume': float(latest_bar['volume']),
            'rsi_14': float(latest_bar.get('rsi_14') or 50.0),
            'macd': float(latest_bar.get('macd') or 0.0),
            'macd_signal': float(latest_bar.get('macd_signal') or 0.0),
            'sma_20': float(latest_bar.get('sma_20') or latest_bar['close_price']),
            'sma_50': float(latest_bar.get('sma_50') or latest_bar['close_price']),
            'volatility_20': float(latest_bar.get('volatility_20') or 0.01),
            'news_sentiment_score': float(mean_sentiment),
            'news_impact_score': 0.5,
            'news_count': len(processed_news),
        }

        # Aplicar el artefacto entrenado
        try:
            artifact = self.predictor.load_model()
            model = artifact['model']
            X_sample = pd.DataFrame([feature_data])[MLTrainer.FEATURE_COLS].astype(float).fillna(0.0)
            probs = model.predict_proba(X_sample)[0]
            prob_map = {str(c): float(p) for c, p in zip(model.classes_, probs)}
            p_sube = prob_map.get('SUBE', 0.33)
            p_estable = prob_map.get('ESTABLE', 0.34)
            p_baja = prob_map.get('BAJA', 0.33)
            total_p = p_sube + p_estable + p_baja
            if total_p > 0:
                p_sube /= total_p
                p_estable /= total_p
                p_baja /= total_p
            predicted_class = max({'SUBE': p_sube, 'ESTABLE': p_estable, 'BAJA': p_baja}, key={'SUBE': p_sube, 'ESTABLE': p_estable, 'BAJA': p_baja}.get)
        except Exception:
            # Si el modelo no ha sido entrenado aún, estimación analítica
            if feature_data['rsi_14'] > 60 or mean_sentiment > 0.2:
                predicted_class, p_sube, p_estable, p_baja = 'SUBE', 0.60, 0.25, 0.15
            elif feature_data['rsi_14'] < 40 or mean_sentiment < -0.2:
                predicted_class, p_sube, p_estable, p_baja = 'BAJA', 0.15, 0.25, 0.60
            else:
                predicted_class, p_sube, p_estable, p_baja = 'ESTABLE', 0.25, 0.50, 0.25

        max_prob = max(p_sube, p_estable, p_baja)

        # 4. Generar Consejo Estratégico
        advice = generate_strategic_advice(
            predicted_class=predicted_class,
            max_prob=max_prob,
            rsi=feature_data['rsi_14'],
            macd=feature_data['macd'],
            sentiment_score=mean_sentiment,
            price=current_price,
            sma20=feature_data['sma_20'],
        )

        # 5. Formatear serie de mercado para gráfico
        market_points = []
        for _, row in df_ind.tail(30).iterrows():
            market_points.append({
                'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                'open_price': round(float(row['open_price']), 2),
                'high_price': round(float(row['high_price']), 2),
                'low_price': round(float(row['low_price']), 2),
                'close_price': round(float(row['close_price']), 2),
                'volume': float(row['volume']),
                'rsi_14': round(float(row['rsi_14']), 1) if pd.notna(row['rsi_14']) else None,
                'macd': round(float(row['macd']), 3) if pd.notna(row['macd']) else None,
                'macd_signal': round(float(row['macd_signal']), 3) if pd.notna(row['macd_signal']) else None,
                'sma_20': round(float(row['sma_20']), 2) if pd.notna(row['sma_20']) else None,
                'sma_50': round(float(row['sma_50']), 2) if pd.notna(row['sma_50']) else None,
            })

        # 6. Guardar en UserScan (con auto-reparación de migración si la tabla aún no existe)
        market_snapshot_data = {
            'rsi_14': round(feature_data['rsi_14'], 1),
            'macd': round(feature_data['macd'], 3),
            'sma_20': round(feature_data['sma_20'], 2),
            'sma_50': round(feature_data['sma_50'], 2),
            'news_sentiment': round(mean_sentiment, 3),
            'news_count': len(processed_news),
        }
        scan_kwargs = dict(
            user=user if (user and user.is_authenticated) else None,
            query=clean_query,
            ticker=ticker_symbol,
            asset_name=name,
            current_price=round(current_price, 2),
            price_change_pct=round(change_pct, 2),
            currency=currency,
            predicted_class=predicted_class,
            prob_sube=round(p_sube, 4),
            prob_estable=round(p_estable, 4),
            prob_baja=round(p_baja, 4),
            advice=advice,
            top_news=processed_news,
            market_snapshot=market_snapshot_data,
        )
        scan_id = 0
        created_at_iso = timezone.now().isoformat()
        try:
            scan_record = UserScan.objects.create(**scan_kwargs)
            scan_id = scan_record.id
            created_at_iso = scan_record.created_at.isoformat()
        except Exception:
            try:
                from django.core.management import call_command
                call_command('migrate', interactive=False, verbosity=0)
                scan_record = UserScan.objects.create(**scan_kwargs)
                scan_id = scan_record.id
                created_at_iso = scan_record.created_at.isoformat()
            except Exception:
                pass

        return {
            'id': scan_id,
            'query': clean_query,
            'keywords': clean_keywords,
            'query_type': query_type,
            'theme_label': theme_label,
            'ticker': ticker_symbol,
            'name': name,
            'current_price': round(current_price, 2),
            'price_change_pct': round(change_pct, 2),
            'currency': currency,
            'prediction': {
                'predicted_class': predicted_class,
                'prob_sube': round(p_sube, 4),
                'prob_estable': round(p_estable, 4),
                'prob_baja': round(p_baja, 4),
                'confidence_pct': round(max_prob * 100, 1),
                'model_version': self.model_version,
            },
            'advice': advice,
            'top_news': processed_news,
            'all_news_count': len(processed_news),
            'market_data': market_points,
            'market_snapshot': market_snapshot_data,
            'created_at': created_at_iso,
        }

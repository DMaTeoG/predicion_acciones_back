import yfinance as yf
import numpy as np
import pandas as pd
from django.utils import timezone
from apps.core.models import Asset, MarketData
from .technical_indicators import TechnicalIndicatorsCalculator


class MarketDataUnavailable(RuntimeError):
    """Raised when yfinance cannot provide usable market data."""


class MarketFetcher:
    def __init__(self):
        self.calculator = TechnicalIndicatorsCalculator()

    def sync_asset_market_data(self, asset: Asset, period: str = "60d", interval: str = "1d") -> int:
        """
        Sincroniza los datos OHLCV de mercado para un activo específico desde yfinance.
        Calcula e inserta/actualiza indicadores técnicos en la BD.
        """
        ticker_symbol = asset.ticker
        records_processed = 0

        try:
            ticker_obj = yf.Ticker(ticker_symbol)
            df = ticker_obj.history(period=period, interval=interval)
        except Exception as exc:
            raise MarketDataUnavailable(
                f"No se pudieron obtener datos de mercado para {ticker_symbol}: {exc}"
            ) from exc

        if df is None or df.empty:
            raise MarketDataUnavailable(
                f"yfinance no devolvió datos de mercado para {ticker_symbol}"
            )

        df = df.reset_index()
        col_map = {
            'Date': 'timestamp',
            'Datetime': 'timestamp',
            'Open': 'open_price',
            'High': 'high_price',
            'Low': 'low_price',
            'Close': 'close_price',
            'Volume': 'volume',
        }
        df = df.rename(columns=col_map)

        required_cols = [
            'timestamp', 'open_price', 'high_price', 'low_price', 'close_price', 'volume'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise MarketDataUnavailable(
                f"yfinance devolvió columnas incompletas para {ticker_symbol}: "
                f"{', '.join(missing_cols)}"
            )

        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        numeric_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Descartar filas incompletas (ej. barra intradía o sesión sin cerrar de yfinance)
        df = df.dropna(subset=required_cols).reset_index(drop=True)

        if df.empty or not np.isfinite(df[numeric_cols].to_numpy(dtype=float)).all():
            raise MarketDataUnavailable(
                f"yfinance devolvió timestamps o valores OHLCV no válidos para {ticker_symbol}"
            )

        price_cols = ['open_price', 'high_price', 'low_price', 'close_price']
        invalid_prices = df[price_cols].le(0).any(axis=1)
        invalid_ranges = (
            (df['high_price'] < df[['open_price', 'close_price']].max(axis=1))
            | (df['low_price'] > df[['open_price', 'close_price']].min(axis=1))
            | df['volume'].lt(0)
        )
        if invalid_prices.any() or invalid_ranges.any():
            raise MarketDataUnavailable(
                f"yfinance devolvió rangos OHLCV imposibles para {ticker_symbol}"
            )

        df = (
            df.sort_values('timestamp')
            .drop_duplicates(subset=['timestamp'], keep='last')
            .reset_index(drop=True)
        )

        # Calcular Indicadores Técnicos
        df = self.calculator.calculate_indicators(df)

        for _, row in df.iterrows():
            ts = row['timestamp']
            if isinstance(ts, (str, pd.Timestamp)):
                ts = pd.to_datetime(ts)
                if ts.tzinfo is None:
                    ts = timezone.make_aware(ts)

            MarketData.objects.update_or_create(
                asset=asset,
                timestamp=ts,
                defaults={
                    'open_price': float(row['open_price']),
                    'high_price': float(row['high_price']),
                    'low_price': float(row['low_price']),
                    'close_price': float(row['close_price']),
                    'volume': float(row['volume']),
                    'rsi_14': float(row['rsi_14']),
                    'macd': float(row['macd']),
                    'macd_signal': float(row['macd_signal']),
                    'sma_20': float(row['sma_20']),
                    'sma_50': float(row['sma_50']),
                    'volatility_20': float(row['volatility_20']),
                }
            )
            records_processed += 1

        return records_processed
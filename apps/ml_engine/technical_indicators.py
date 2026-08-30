import pandas as pd
import numpy as np

class TechnicalIndicatorsCalculator:
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula indicadores técnicos sobre un DataFrame con columnas OHLCV:
        ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
        Devuelve el DataFrame enriquecido con las columnas de indicadores.
        """
        if df.empty or len(df) < 5:
            df['rsi_14'] = 50.0
            df['macd'] = 0.0
            df['macd_signal'] = 0.0
            df['sma_20'] = df['close_price'] if not df.empty else 0.0
            df['sma_50'] = df['close_price'] if not df.empty else 0.0
            df['volatility_20'] = 0.01
            return df

        df = df.sort_values('timestamp').reset_index(drop=True)
        close = df['close_price']

        # 1. Medias Móviles Simples (SMA 20 y SMA 50)
        df['sma_20'] = close.rolling(window=20, min_periods=1).mean()
        df['sma_50'] = close.rolling(window=50, min_periods=1).mean()

        # 2. RSI (14 periodos)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_14'] = df['rsi_14'].fillna(50.0)

        # 3. MACD y MACD Signal (12, 26, 9)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # 4. Volatilidad (Desviación estándar móvil de 20 periodos de retornos porcentuales)
        pct_change = close.pct_change().fillna(0.0)
        df['volatility_20'] = pct_change.rolling(window=20, min_periods=1).std().fillna(0.01)

        return df

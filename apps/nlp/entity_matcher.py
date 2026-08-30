import re
from typing import Optional

class EntityMatcher:
    def __init__(self, asset_model):
        self.asset_model = asset_model
        # Mapeos predeterminados de alias y palabras clave a tickers
        self.alias_map = {
            'APPLE': 'AAPL',
            'AAPL': 'AAPL',
            'TESLA': 'TSLA',
            'TSLA': 'TSLA',
            'MICROSOFT': 'MSFT',
            'MSFT': 'MSFT',
            'NVIDIA': 'NVDA',
            'NVDA': 'NVDA',
            'BITCOIN': 'BTC-USD',
            'BTC': 'BTC-USD',
            'ETHEREUM': 'ETH-USD',
            'ETH': 'ETH-USD',
        }

    def match_asset(self, text: str):
        """
        Busca menciones de activos financieros en el texto provisto.
        Devuelve el objeto Asset coincidente o None.
        """
        if not text:
            return None

        upper_text = text.upper()

        # Buscar coincidencia exacta por palabras
        words = set(re.findall(r'\b[A-Z0-9\-]+\b', upper_text))

        matched_ticker = None
        for word in words:
            if word in self.alias_map:
                matched_ticker = self.alias_map[word]
                break

        if matched_ticker:
            try:
                return self.asset_model.objects.get(ticker=matched_ticker, is_active=True)
            except self.asset_model.DoesNotExist:
                return None

        # Si no hay coincidencia exacta de palabra, probar coincidencia de subcadena
        active_assets = self.asset_model.objects.filter(is_active=True)
        for asset in active_assets:
            if asset.name.upper() in upper_text or asset.ticker.upper() in upper_text:
                return asset

        return None

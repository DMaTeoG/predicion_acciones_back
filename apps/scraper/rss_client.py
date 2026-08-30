from datetime import timezone as dt_timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlencode
from xml.etree import ElementTree as ET
import requests


class RSSFeedError(RuntimeError):
    """Raised when a news feed cannot be fetched or parsed safely."""


def parse_rss_items(xml_bytes: bytes, ticker: str) -> list[dict]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise RSSFeedError(f"El feed RSS de {ticker} contiene XML inválido") from exc

    articles = []
    for item in root.findall('./channel/item'):
        title = (item.findtext('title') or '').strip()
        url = (item.findtext('link') or '').strip()
        pub_date = (item.findtext('pubDate') or '').strip()
        if not title or not url or not pub_date:
            continue

        try:
            published_at = parsedate_to_datetime(pub_date)
        except (TypeError, ValueError) as exc:
            raise RSSFeedError(
                f"El feed RSS de {ticker} contiene una fecha de publicación inválida"
            ) from exc
        if published_at is None:
            raise RSSFeedError(
                f"El feed RSS de {ticker} contiene una fecha de publicación inválida"
            )
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=dt_timezone.utc)

        source_node = item.find('source')
        source = (source_node.text or '').strip() if source_node is not None else ''
        articles.append({
            'ticker': ticker,
            'title': title,
            'summary': (item.findtext('description') or '').strip(),
            'url': url,
            'source': source or 'Google News RSS',
            'published_at': published_at,
        })

    return articles


class GoogleNewsRssClient:
    FEED_URL = 'https://news.google.com/rss/search'

    def build_feed_url(self, asset) -> str:
        query = f'{asset.ticker} OR "{asset.name}"'
        params = {
            'q': query,
            'hl': 'en-US',
            'gl': 'US',
            'ceid': 'US:en',
        }
        return f'{self.FEED_URL}?{urlencode(params)}'

    def fetch(self, asset) -> list[dict]:
        feed_url = self.build_feed_url(asset)
        try:
            response = requests.get(feed_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RSSFeedError(
                f"No se pudo obtener el feed de noticias para {asset.ticker}: {exc}"
            ) from exc

        return parse_rss_items(response.content, asset.ticker)

    def fetch_by_query(self, query: str, ticker: str = "") -> list[dict]:
        """Permite buscar noticias por palabras clave directas o temas macroeconómicos."""
        params = {
            'q': query,
            'hl': 'en-US',
            'gl': 'US',
            'ceid': 'US:en',
        }
        feed_url = f'{self.FEED_URL}?{urlencode(params)}'
        try:
            response = requests.get(feed_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RSSFeedError(f"No se pudo obtener el feed para la consulta '{query}': {exc}") from exc

        return parse_rss_items(response.content, ticker or query)

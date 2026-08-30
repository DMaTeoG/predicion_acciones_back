import scrapy
from datetime import datetime
from django.utils import timezone

class FinancialNewsSpider(scrapy.Spider):
    name = 'financial_news'
    allowed_domains = ['finance.yahoo.com', 'investing.com', 'reuters.com']
    start_urls = [
        'https://finance.yahoo.com/news/',
    ]

    custom_settings = {
        'ITEM_PIPELINES': {
            'apps.scraper.pipelines.DjangoNewsPipeline': 300,
        },
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }

    def parse(self, response):
        articles = response.css('article') or response.css('div.news-link') or response.css('li.stream-item')

        for article in articles:
            title = article.css('h3::text, a::text').get()
            url = article.css('a::attr(href)').get()
            summary = article.css('p::text').get() or ''

            if title and url:
                full_url = response.urljoin(url)
                yield {
                    'title': title.strip(),
                    'summary': summary.strip(),
                    'url': full_url,
                    'source': 'Yahoo Finance News',
                    'published_at': timezone.now(),
                }

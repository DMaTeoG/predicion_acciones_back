class DjangoNewsPipeline:
    def __init__(self):
        from apps.core.models import Asset, NewsArticle, NewsSentiment
        from apps.nlp.sentiment_engine import SentimentEngine
        from apps.nlp.entity_matcher import EntityMatcher
        
        self.Asset = Asset
        self.NewsArticle = NewsArticle
        self.NewsSentiment = NewsSentiment
        self.sentiment_engine = SentimentEngine()
        self.entity_matcher = EntityMatcher(Asset)

    def process_item(self, item, spider):
        title = item.get('title', '').strip()
        summary = item.get('summary', '').strip()
        url = item.get('url', '').strip()
        source = item.get('source', 'Financial Web Scraper').strip()
        published_at = item.get('published_at')

        if not title or not url or published_at is None:
            return item

        requested_ticker = item.get('ticker')
        if requested_ticker:
            asset = self.Asset.objects.filter(
                ticker__iexact=requested_ticker,
                is_active=True,
            ).first()
        else:
            text_for_matching = f"{title} {summary}"
            asset = self.entity_matcher.match_asset(text_for_matching)

        if not asset:
            return item

        # Save NewsArticle if URL does not exist
        article, created = self.NewsArticle.objects.get_or_create(
            url=url,
            defaults={
                'asset': asset,
                'title': title,
                'summary': summary,
                'content': item.get('content', ''),
                'source': source,
                'published_at': published_at
            }
        )

        if created or not hasattr(article, 'sentiment'):
            # Analyze Sentiment with NLP engine
            nlp_result = self.sentiment_engine.analyze_text(f"{title} {summary}")

            self.NewsSentiment.objects.update_or_create(
                article=article,
                defaults={
                    'sentiment_label': nlp_result['sentiment_label'],
                    'sentiment_score': nlp_result['sentiment_score'],
                    'impact_score': nlp_result['impact_score']
                }
            )

        return item

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentEngine:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_text(self, text: str) -> dict:
        """
        Analiza el texto de una noticia y calcula el puntaje continuo (-1.0 a 1.0)
        así como la etiqueta discreta (POSITIVE, NEUTRAL, NEGATIVE) e impacto estimado.
        """
        if not text or not text.strip():
            return {
                'sentiment_label': 'NEUTRAL',
                'sentiment_score': 0.0,
                'impact_score': 0.5
            }

        vs = self.analyzer.polarity_scores(text)
        compound = vs['compound']  # Rango de -1.0 a 1.0

        if compound >= 0.05:
            label = 'POSITIVE'
        elif compound <= -0.05:
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'

        # El impacto se calcula como el valor absoluto de la intensidad del sentimiento
        impact = min(1.0, max(0.2, abs(compound) + 0.2))

        return {
            'sentiment_label': label,
            'sentiment_score': float(compound),
            'impact_score': float(impact)
        }
